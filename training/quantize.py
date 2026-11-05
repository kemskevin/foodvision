"""Quantization INT8 d'un modèle ONNX avec ONNX Runtime.

- quantize_int8 : quantization statique (format QDQ) avec calibration.
  op_types permet de ne quantiser que certains opérateurs (ex : Conv/MatMul/Gemm)
  et de laisser en FP32 les opérateurs sensibles (Sigmoid, Mul des blocs squeeze-excitation).
  reduce_range=True évite la saturation sur les CPU x86 sans instructions VNNI.
- quantize_dynamic_int8 : quantization dynamique (poids seulement, sans calibration).
"""
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


class ListCalibrationReader(CalibrationDataReader):
    def __init__(self, arrays, input_name: str = "input"):
        self._iterator = iter([{input_name: a} for a in arrays])

    def get_next(self):
        return next(self._iterator, None)


def _preprocess(src: Path) -> Path:
    pre = src.with_name(src.stem + "_pre.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(src), str(pre))
        return pre
    except Exception as exc:  # le pré-traitement est optionnel
        print(f"[quantize] pré-traitement ignoré : {exc}")
        return src


def quantize_int8(fp32_path, int8_path, calibration_arrays, method: str = "minmax",
                  op_types=None, reduce_range: bool = True):
    src = Path(fp32_path)
    model_in = _preprocess(src)
    calib_method = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }[method]
    quantize_static(
        str(model_in),
        str(int8_path),
        ListCalibrationReader(calibration_arrays),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=reduce_range,
        op_types_to_quantize=op_types,
        calibrate_method=calib_method,
    )
    if model_in != src and model_in.exists():
        model_in.unlink()
    print(f"[quantize] statique ({method}, ops={op_types or 'tous'}) -> {int8_path}")


def quantize_dynamic_int8(fp32_path, int8_path):
    src = Path(fp32_path)
    model_in = _preprocess(src)
    quantize_dynamic(str(model_in), str(int8_path), weight_type=QuantType.QUInt8)
    if model_in != src and model_in.exists():
        model_in.unlink()
    print(f"[quantize] dynamique -> {int8_path}")
