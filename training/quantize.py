"""Quantization statique INT8 (format QDQ) d'un modèle ONNX avec ONNX Runtime."""
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class ListCalibrationReader(CalibrationDataReader):
    def __init__(self, arrays, input_name: str = "input"):
        self._iterator = iter([{input_name: a} for a in arrays])

    def get_next(self):
        return next(self._iterator, None)


def quantize_int8(fp32_path, int8_path, calibration_arrays, method: str = "minmax"):
    src = Path(fp32_path)
    pre = src.with_name(src.stem + "_pre.onnx")
    model_in = src
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(src), str(pre))
        model_in = pre
    except Exception as exc:  # le pré-traitement est optionnel
        print(f"[quantize] pré-traitement ignoré : {exc}")

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
        calibrate_method=calib_method,
    )
    if pre.exists():
        pre.unlink()
    print(f"[quantize] modèle INT8 écrit dans {int8_path}")
