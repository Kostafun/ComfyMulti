INPUT_SCHEMA = {
    'source_video': {
        'type': str,
        'required': True
    },
    'source_audio': {
        'type': str,
        'required': True
    },
    'face_restore': {
        'type': bool,
        'required': False,
        'default': False
    },
    'upscale': {
        'type': int,
        'required': False,
        'default': 1
    },
    'codeformer_fidelity': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'start_frame': {
        'type': int,
        'required': False,
        'default': 0
    }
    # 'output_format': {
    #     'type': str,
    #     'required': False,
    #     'default': 'JPEG',
    #     'constraints': lambda output_format: output_format in [
    #         'JPEG',
    #         'PNG'
    #     ]
    # }
}
