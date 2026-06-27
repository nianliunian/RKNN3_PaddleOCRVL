import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/vision/PP-DocLayoutV2_simplified.onnx'

RKNN_MODEL = '../../model/vision/PP-DocLayoutV2.rknn'
TARGET_PLATFORM = 'rk1820'

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="PP-DocLayoutV2 ONNX to RKNN converter")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="ONNX model path")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="Output RKNN model path")
    parser.add_argument("--target_platform", type=str, default=TARGET_PLATFORM, help="Target platform")
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> Config model')
    rknn.config(target_platform=args.target_platform, #quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
                optimization_level=0)
    print('done')
    

    # Load ONNX model — fix dynamic dims with static shapes (batch=1)
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path,
                         inputs=['im_shape', 'image', 'scale_factor'],
                         input_size_list=[[1, 2], [1, 3, 800, 800], [1, 2]])
    
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
 
    # Build model (no quantization)
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    rknn.release()
    print('DONE')
    exit(-1)
    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
    print('DONE')
