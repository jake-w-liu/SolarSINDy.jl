"""Independent data-only checker for the locked mixed8 ridge model codec.

This checker does not fit a model, execute Julia, load observations, establish
archival authority or certify a scientific result. It is not the Julia fitter.
"""
from pathlib import Path
from hashlib import sha256
from typing import NamedTuple
import argparse
import importlib.util
import math
import struct

HELPER = Path(__file__).resolve().with_name('bounded_json.py')
with HELPER.open('rb') as stream:
    helper_bytes = stream.read(262145)
if len(helper_bytes) > 262144 or sha256(helper_bytes).hexdigest() != '1912a60ab9e014437f54ae133caec353fcf2839972344a126b27836b23f0aa5f':
    raise RuntimeError('bounded JSON checker source differs')
spec = importlib.util.spec_from_file_location('locked_ridge_bounded_json', HELPER)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
need = V.need
FAMILY = 'locked_mixed8_ridge_log1p_v1'
RECIPE_SCHEMA = 'archive_locked_mixed8_ridge_recipe_v1'
MODEL_SCHEMA = 'archive_locked_mixed8_ridge_model_v1'
FEATURES = ('dbdt_now', 'dbdt_mean30', 'dbdt_max30', 'dbdt_std30', 'V', 'Bz', 'Bs', 'VBs')
MAX_BYTES = 16384
MAX_ROWS = 600000


def bits(value):
    return struct.pack('>d', value).hex()


def number(value):
    need(isinstance(value, str) and len(value) == 16 and
         all(c in '0123456789abcdef' for c in value), 'locked ridge raw-bit syntax')
    result = struct.unpack('>d', bytes.fromhex(value))[0]
    need(math.isfinite(result), 'locked ridge nonfinite encoded number')
    return result


def recipe_object():
    # Literal protocol constants, independent of the Julia recipe constructor.
    return dict(schema=RECIPE_SCHEMA, family=FAMILY, layout='mixed8',
        feature_names=list(FEATURES), target_transform='log1p_rate_reference_1_nt_per_min',
        penalty='3ff0000000000000', intercept_penalty=True,
        std_rule='sample_std_plus_1e_minus9', epsilon='3e112e0be826d695')


class Model(NamedTuple):
    center: tuple
    scale: tuple
    coefficients: tuple
    model_sha256: str
    recipe_sha256: str


def _locked(value, data, model_pin, recipe_pin):
    need(isinstance(value, dict) and set(value) ==
         {'schema', 'family', 'mode', 'recipe', 'center', 'scale', 'coefficients'},
         'locked ridge model fields')
    need(value['schema'] == MODEL_SCHEMA and value['family'] == FAMILY and value['mode'] == 'mixed8',
         'locked ridge model convention')
    expected_recipe = V.canonical(recipe_object())
    need(V.canonical(value['recipe']) == expected_recipe and
         sha256(expected_recipe).hexdigest() == recipe_pin, 'locked ridge recipe binding differs')
    parsed = []
    for name, width in (('center', 8), ('scale', 8), ('coefficients', 9)):
        row = value[name]
        need(isinstance(row, list) and len(row) == width, 'locked ridge model geometry')
        parsed.append(tuple(number(item) for item in row))
    need(all(item >= 1e-9 for item in parsed[1]), 'locked ridge model scale domain')
    return Model(*parsed, model_pin, recipe_pin)


# Explicit schema registration: no model family falls through to another codec.
MODEL_READERS = {MODEL_SCHEMA: _locked}


def decode_model(data, *, model_sha256, recipe_sha256):
    V._sha(model_sha256)
    V._sha(recipe_sha256)
    need(isinstance(data, bytes) and len(data) <= MAX_BYTES, 'locked ridge model byte bound')
    need(sha256(data).hexdigest() == model_sha256, 'external locked ridge model pin differs')
    value = V.decoded(data)
    need(isinstance(value, dict) and value.get('schema') in MODEL_READERS, 'unsupported numeric model schema')
    return MODEL_READERS[value['schema']](value, data, model_sha256, recipe_sha256)


def read_model(path, *, model_sha256, recipe_sha256):
    path = V._ordinary(Path(path))
    for part in (path, *path.parents):
        need(not getattr(part.lstat(), 'st_file_attributes', 0) & 0x400, 'reparse model path component')
    return decode_model(V.bounded_read(path, MAX_BYTES), model_sha256=model_sha256, recipe_sha256=recipe_sha256)


def predict_fixture(model, features):
    """Independent bounded fixture calculation; not a production fitting backend."""
    need(isinstance(model, Model) and isinstance(features, (list, tuple)) and len(features) <= MAX_ROWS,
         'locked ridge fixture rows')
    result = []
    for row in features:
        need(isinstance(row, (list, tuple)) and len(row) == 8, 'locked ridge fixture feature width')
        values = []
        for item in row:
            need(isinstance(item, (int, float)) and not isinstance(item, bool), 'locked ridge fixture number type')
            try:
                value = float(item)
            except (OverflowError, ValueError) as error:
                raise V.SnapshotError('locked ridge fixture Float64 range') from error
            need(math.isfinite(value), 'locked ridge fixture finite number')
            values.append(value)
        terms = [model.coefficients[0]]
        for value, center, scale, coefficient in zip(values, model.center, model.scale, model.coefficients[1:]):
            transformed = (value-center)/scale
            term = transformed*coefficient
            need(math.isfinite(transformed) and math.isfinite(term), 'locked ridge fixture arithmetic overflow')
            terms.append(term)
        try:
            prediction = math.fsum(terms)
        except (OverflowError, ValueError) as error:
            raise V.SnapshotError('locked ridge fixture reduction overflow') from error
        need(math.isfinite(prediction), 'locked ridge fixture prediction overflow')
        result.append(prediction)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--recipe-sha256', required=True)
    args = parser.parse_args()
    model = read_model(args.model, model_sha256=args.model_sha256, recipe_sha256=args.recipe_sha256)
    print(V.canonical(dict(complete_as_data_only_codec_check=True,
        schema=MODEL_SCHEMA, model_sha256=model.model_sha256, recipe_sha256=model.recipe_sha256,
        actual_source_authority_granted=False, actual_fits=0, native_Julia_parity_verified=False,
        scientific_qualification=False)).decode(), end='')


if __name__ == '__main__':
    main()
