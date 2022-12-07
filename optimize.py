import logging
from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser

from model.inference import InferenceBinder


@dataclass
class OptimizationArguments:
    model: Path = field(metadata={'help': 'Model .pkl file to optimize'})

    prune: float = field(default=0.0, metadata={'help': 'Fraction of all heads to prune.'})
    prune_iterations: int = field(default=5, metadata={'help': 'Pruning iterations (the higher the better)'})
    batch_size: int = field(default=1, metadata={'help': 'Batch size for head importances evaluation.'})
    dataset_dir: Path = field(default=Path('data'), metadata={'help': 'Dataset to use for pruning.'})

    onnx_dir: Path = field(default=Path('onnx'), metadata={'help': 'Path to directory where to store ONNX models.'})
    fuse: bool = field(default=False, metadata={'help': 'Fuse some elements of the model (is not supported with quantization)'})
    quantize: bool = field(default=False, metadata={'help': 'Quantize the model.'})
    opset_version: int = field(default=13, metadata={'help': 'ONNX opset version: 11, 12 or 13.'})
    do_constant_folding: bool = field(default=False, metadata={'help': 'Fold constants during ONNX conversion.'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = HfArgumentParser(dataclass_types=[OptimizationArguments])
    opt_args, = parser.parse_args_into_dataclasses()
    opt_args: OptimizationArguments

    model = InferenceBinder.load(opt_args.model)

    if opt_args.prune > 0:
        model.prune(opt_args.dataset_dir, opt_args.prune, opt_args.prune_iterations, opt_args.batch_size)
        model.cpu()
        pruned_path = Path(opt_args.model.parent.joinpath('pruned_' + opt_args.model.name))
        model.save(pruned_path)
        logging.info(f'Model pruned and saved to {pruned_path}')

    model.optimize(
        opt_args.onnx_dir,
        quant=opt_args.quantize,
        fuse=opt_args.fuse,
        opset_version=opt_args.opset_version,
        do_constant_folding=opt_args.do_constant_folding
    )
    opt_path = opt_args.onnx_dir.joinpath('main.pkl')
    model.save(opt_path)
    logging.info(f'Model optimized and saved to {opt_path}')
