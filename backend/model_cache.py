import os

model_info_cache = dict()

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def get_model_path(model_role, language):
    return os.path.join(BASE_DIR, language, f'{model_role}_model.bin')

def get_model_path_new(model_role, language):
    return os.path.join(BASE_DIR, language, model_role, f'pytorch_model.bin')

def load_model_with_cache(model_role, language, model_loader):
    '''`model_loader` should return (model, tokenizer, device).'''
    try:
        model_info = model_info_cache[model_role][language]
    except Exception as err:
        print(f"+++ Model type: {model_role} is not loaded for language: {language}. Trying to load model...")
        model_info = model_loader(get_model_path_new(model_role, language))

        if model_role not in model_info_cache:
            model_info_cache[model_role] = dict()
        model_info_cache[model_role][language] = model_info

        print(f"+++ Model type: {model_role} for language: {language} is loaded")
    return model_info
