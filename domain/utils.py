def trim_base64_header(logger, data:str):
    try:
        if data.startswith('data:'):
            ext = data.split(';')[0][11:]
            data = data.split('base64,')[1]
            return data, ext
        else:
            return data, 'jpg'
    except Exception as e:
        logger.error(e)
        return None