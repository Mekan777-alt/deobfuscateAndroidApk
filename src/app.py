import os
import asyncio
import aiofiles
import glob
import logging
import json
import time
from openai import AsyncOpenAI
from core.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('deobfuscator')
file_handler = None  # Будет создан после инициализации директории вывода


class CodeDeobfuscator:
    """Класс для деобфускации Java/Kotlin кода"""

    def __init__(self):
        """Инициализация деобфускатора"""
        self.setup_directories()
        self.prompt_template = ""
        self.processed_files = set()
        self.progress_file = os.path.join(settings.app.absolute_output_dir, "progress.json")

        # Создаем клиент OpenAI
        self.client = AsyncOpenAI(
            api_key=settings.openai.api_key,
            timeout=settings.openai.request_timeout
        )

        # Добавляем файловый обработчик логов после создания директории вывода
        global file_handler
        if file_handler is None:
            log_file = os.path.join(settings.app.absolute_output_dir, 'deobfuscator.log')
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    def setup_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(settings.app.absolute_output_dir, exist_ok=True)
        os.makedirs(settings.app.absolute_prompts_dir, exist_ok=True)

    async def read_prompt_file(self):
        """Асинхронно читает файл промпта из prompts/deobfuscate_prompt.txt"""
        prompt_file = os.path.join(settings.app.absolute_prompts_dir, 'deobfuscate_prompt.txt')

        async with aiofiles.open(prompt_file, 'r') as f:
            self.prompt_template = await f.read()

        logger.info("Промпт успешно загружен из файла")

    def load_progress(self):
        """Загружает сохраненный прогресс обработки"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.processed_files = set(progress_data.get("processed_files", []))
                    logger.info(f"Загружен прогресс: обработано {len(self.processed_files)} файлов")
                    return True
            except Exception as e:
                logger.error(f"Ошибка при загрузке прогресса: {e}")
        return False

    def save_progress(self):
        """Сохраняет текущий прогресс обработки"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({"processed_files": list(self.processed_files)}, f)
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогресса: {e}")
            return False

    async def send_to_openai(self, code):
        """Асинхронно отправляет код в OpenAI для деобфускации используя новый API"""
        if not settings.openai.api_key:
            logger.error("API ключ OpenAI не установлен")
            return None

        prompt = self.prompt_template.replace("{code}", code)

        retries = 3
        delay = 2

        for attempt in range(retries):
            try:
                chat_completion = await self.client.chat.completions.create(
                    model=settings.openai.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.openai.temperature,
                )

                if chat_completion.choices and len(chat_completion.choices) > 0:
                    response_text = chat_completion.choices[0].message.content

                    # Извлекаем JSON из ответа
                    try:
                        # Поиск JSON-блока в ответе
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1

                        if json_start >= 0 and json_end > json_start:
                            json_text = response_text[json_start:json_end]
                            response_data = json.loads(json_text)

                            if "Code" in response_data:
                                return response_data["Code"]
                            else:
                                logger.warning("В ответе API отсутствует ключ 'Code'")
                                # Возвращаем весь текст как запасной вариант
                                return response_text.strip()
                        else:
                            logger.warning("Не найден JSON в ответе API")
                            # Возвращаем весь текст как запасной вариант
                            return response_text.strip()

                    except json.JSONDecodeError:
                        logger.warning("Ошибка декодирования JSON, возвращаем оригинальный ответ")
                        # Запасной вариант - возвращаем весь текст без обработки JSON
                        return response_text.strip()
                else:
                    logger.error("Некорректный ответ от API")
                    return None

            except Exception as e:
                logger.error(f"Ошибка при запросе к API (попытка {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info(f"Повтор через {delay} секунд...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Увеличиваем задержку экспоненциально
                else:
                    logger.error("Все попытки запроса исчерпаны")

        return None

    async def process_file(self, semaphore, file_path):
        """Асинхронно обрабатывает один файл: читает, деобфусцирует и сохраняет"""
        async with semaphore:
            try:
                # Получаем относительный путь для сохранения структуры директорий
                rel_path = os.path.relpath(file_path, settings.app.absolute_input_dir)
                output_file = os.path.join(settings.app.absolute_output_dir, rel_path)

                # Проверяем, был ли файл уже обработан
                if rel_path in self.processed_files:
                    logger.info(f"Пропуск уже обработанного файла: {rel_path}")
                    return True

                logger.info(f"Обработка: {rel_path}")

                # Создаем директорию если не существует
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Асинхронно читаем файл
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        code = await f.read()
                except UnicodeDecodeError:
                    # Пробуем другую кодировку если utf-8 не сработал
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                            code = await f.read()
                    except Exception as e:
                        logger.error(f"Не удалось прочитать файл {file_path}: {e}")
                        return False

                # Пропускаем пустые файлы
                if not code.strip():
                    logger.info(f"Пропуск пустого файла: {rel_path}")
                    self.processed_files.add(rel_path)
                    return False

                # Отправляем на деобфускацию
                deobfuscated_code = await self.send_to_openai(code)

                if deobfuscated_code:
                    # Асинхронно сохраняем результат
                    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                        await f.write(deobfuscated_code)
                    logger.info(f"Деобфусцировано и сохранено: {rel_path}")
                    self.processed_files.add(rel_path)
                    return True
                else:
                    logger.error(f"Не удалось деобфусцировать: {rel_path}")
                    # Копируем оригинальный файл
                    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                        await f.write(code)
                    logger.info(f"Скопирован оригинальный файл: {rel_path}")
                    self.processed_files.add(rel_path)
                    return False

            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")
                return False

    async def find_code_files(self, directory):
        """Асинхронно находит все Java и Kotlin файлы в указанной директории"""
        code_files = []

        # Паттерны для поиска Java и Kotlin файлов
        java_pattern = os.path.join(directory, "**", "*.java")
        kotlin_pattern = os.path.join(directory, "**", "*.kt")

        # Ищем файлы асинхронно
        loop = asyncio.get_event_loop()
        java_files = await loop.run_in_executor(None, lambda p: glob.glob(p, recursive=True), java_pattern)
        kotlin_files = await loop.run_in_executor(None, lambda p: glob.glob(p, recursive=True), kotlin_pattern)

        code_files = java_files + kotlin_files
        logger.info(f"Найдено {len(code_files)} исходных файлов")

        return code_files

    async def run(self):
        """Запускает процесс деобфускации всех исходных файлов"""
        logger.info("Запуск деобфускатора")

        # Проверяем API ключ
        if not settings.openai.api_key:
            logger.error("OPENAI_API_KEY не задан. Установите переменную окружения.")
            return False

        # Проверяем, существует ли директория с исходным кодом
        if not os.path.exists(settings.app.absolute_input_dir):
            logger.error(f"Директория с исходным кодом не найдена: {settings.app.absolute_input_dir}")
            return False

        # Читаем файл промпта
        await self.read_prompt_file()

        # Загружаем сохраненный прогресс
        self.load_progress()

        # Находим все исходные файлы
        code_files = await self.find_code_files(settings.app.absolute_input_dir)

        if not code_files:
            logger.error(f"Java/Kotlin файлы не найдены в {settings.app.absolute_input_dir}")
            return False

        # Фильтруем файлы, которые уже были обработаны
        remaining_files = [f for f in code_files if
                           os.path.relpath(f, settings.app.absolute_input_dir) not in self.processed_files]
        logger.info(f"Осталось обработать {len(remaining_files)} из {len(code_files)} файлов")

        # Рассчитываем оптимальное количество потоков
        # Ограничиваем количество параллельных запросов, чтобы не превышать лимиты API
        effective_thread_count = min(settings.app.thread_count, 3)
        semaphore = asyncio.Semaphore(effective_thread_count)

        # Обрабатываем файлы пакетами для контроля нагрузки на API
        batch_size = 10
        save_interval = 20  # Сохранять прогресс каждые N файлов
        files_since_save = 0
        last_batch_time = time.time()

        for i in range(0, len(remaining_files), batch_size):
            batch = remaining_files[i:i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (len(remaining_files) + batch_size - 1) // batch_size

            logger.info(f"Обработка пакета {current_batch} из {total_batches} ({len(batch)} файлов)")

            # Создаем и запускаем задачи для обработки файлов текущего пакета
            tasks = [self.process_file(semaphore, file) for file in batch]
            batch_results = await asyncio.gather(*tasks)

            # Обновляем статистику
            success_count = sum(1 for r in batch_results if r)
            logger.info(f"Пакет {current_batch}: обработано успешно {success_count} из {len(batch)} файлов")

            # Сохраняем прогресс периодически
            files_since_save += len(batch)
            if files_since_save >= save_interval:
                self.save_progress()
                files_since_save = 0
                logger.info(f"Прогресс сохранен: обработано {len(self.processed_files)} файлов")

            # Управление скоростью запросов - добавляем паузу если нужно
            batch_time = time.time() - last_batch_time
            if batch_time < 5 and i + batch_size < len(remaining_files):  # Минимальное время между пакетами - 5 секунд
                pause_time = 5 - batch_time
                logger.info(f"Пауза {pause_time:.2f} секунд для соблюдения лимитов API...")
                await asyncio.sleep(pause_time)

            last_batch_time = time.time()

        # Сохраняем итоговый прогресс
        self.save_progress()

        # Подсчет итоговых результатов
        total_processed = len(self.processed_files)
        success_count = len([f for f in os.listdir(settings.app.absolute_output_dir)
                             if os.path.isfile(os.path.join(settings.app.absolute_output_dir, f))])
        failed_count = total_processed - success_count

        logger.info("Деобфускация завершена")
        logger.info(f"Всего обработано: {total_processed} файлов")
        logger.info(f"Успешно деобфусцировано: {success_count} файлов")
        logger.info(f"Не удалось обработать: {failed_count} файлов")
        logger.info(f"Результаты доступны в {settings.app.absolute_output_dir}")

        return True


async def main():
    """Основная функция запуска программы"""
    deobfuscator = CodeDeobfuscator()
    await deobfuscator.run()


if __name__ == "__main__":
    asyncio.run(main())