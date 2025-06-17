import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

model_info_cache = dict()
model_locks = dict()
executor = ThreadPoolExecutor()
CUDA_ID = os.environ.get("CUDA_ID", 0)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


def get_model_path(model_role, language):
    return os.path.join(BASE_DIR, language, f'{model_role}_model.bin')


async def load_model_with_cache(model_role, language, model_loader):
    if model_role not in model_locks:
        model_locks[model_role] = dict()
    if language not in model_locks[model_role]:
        model_locks[model_role][language] = asyncio.Lock()

    async with model_locks[model_role][language]:
        try:
            model_info = model_info_cache[model_role][language]
        except KeyError:
            print(f"+++ Model type: {model_role} is not loaded for language: {language}. Trying to load model...")
            model_info = await asyncio.get_event_loop().run_in_executor(
                executor, model_loader, get_model_path(model_role, language), CUDA_ID)

            if model_role not in model_info_cache:
                model_info_cache[model_role] = dict()
            model_info_cache[model_role][language] = model_info

            print(f"+++ Model type: {model_role} for language: {language} is loaded")
        return model_info