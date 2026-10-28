"""Literal independent model-codec controls; no Julia or observation access."""
from pathlib import Path
from fractions import Fraction
from hashlib import sha256
import copy
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_locked_mixed8_checker', ROOT/'tools/point_input_export/locked_mixed8_ridge.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture(center=None, scale=None, coefficients=None):
    # The literal family, shape and raw-bit fixture does not use a Julia encoder.
    recipe = dict(schema='archive_locked_mixed8_ridge_recipe_v1', family='locked_mixed8_ridge_log1p_v1',
        layout='mixed8', feature_names=['dbdt_now','dbdt_mean30','dbdt_max30','dbdt_std30','V','Bz','Bs','VBs'],
        target_transform='log1p_rate_reference_1_nt_per_min', penalty='3ff0000000000000',
        intercept_penalty=True, std_rule='sample_std_plus_1e_minus9', epsilon='3e112e0be826d695')
    value = dict(schema='archive_locked_mixed8_ridge_model_v1', family='locked_mixed8_ridge_log1p_v1', mode='mixed8',
        recipe=recipe, center=center or ['0000000000000000']*8,
        scale=scale or ['3e112e0be826d695']*8,
        coefficients=coefficients or ['3fe5555555555555']+['0000000000000000']*8)
    return value


def decode(value):
    body = M.V.canonical(value)
    return M.decode_model(body, model_sha256=sha256(body).hexdigest(), recipe_sha256=sha256(M.V.canonical(value['recipe'])).hexdigest())


class LockedMixed8ReaderTests(unittest.TestCase):
    def test_literal_zero_design_model_intercept_two_thirds(self):
        model = decode(fixture())
        self.assertEqual(model.coefficients, (2/3,)+(0.0,)*8)
        self.assertEqual(model.scale, (1e-9,)*8)
        self.assertEqual(M.predict_fixture(model, [[0.0]*8,[0.0]*8]), [2/3,2/3])
        self.assertEqual(M.predict_fixture(model, []), [])
        self.assertIsInstance(model.coefficients, tuple)

    def test_nonconstant_exact_rational_oracle_and_sample_epsilon(self):
        sigma = Fraction(1000000001,1000000000)
        slope = 2*sigma/(2+sigma*sigma)
        value = fixture(scale=[M.bits(1.0+1e-9)]+['3e112e0be826d695']*7,
            coefficients=[M.bits(3/4),M.bits(float(slope))]+['0000000000000000']*7)
        model = decode(value)
        rows = [[float(x)]+[0.0]*7 for x in (-1,0,1)]
        predictions = M.predict_fixture(model, rows)
        expected = [Fraction(3,4)-2/(2+sigma*sigma), Fraction(3,4), Fraction(3,4)+2/(2+sigma*sigma)]
        for actual, target in zip(predictions, expected):
            # Unit-scale, well-conditioned nine-coefficient fixture; catches epsilon/penalty changes.
            self.assertAlmostEqual(actual, float(target), delta=1e-14)
        self.assertEqual(model.coefficients[0], .75)
        self.assertNotEqual(model.scale[0], 1.0)

    def test_self_consistently_rehashed_wrong_conventions_are_rejected(self):
        variants = [('schema','archive_classic_point_model_v1'),('family','retained_ridge_log1p'),('mode','ground11')]
        for key, wrong in variants:
            with self.subTest(key=key):
                value=fixture();value[key]=wrong
                with self.assertRaises(M.V.SnapshotError): decode(value)
        for key, wrong in [('intercept_penalty',False),('intercept_penalty',1),('penalty','0000000000000000'),
                ('std_rule','sample_std_zero_nonfinite_to_one'),('epsilon','3ff0000000000000'),
                ('layout','ground11'),('target_transform','raw_target'),('feature_names',list(reversed(M.FEATURES)))]:
            with self.subTest(key=key, wrong=wrong):
                value=fixture();value['recipe'][key]=wrong
                with self.assertRaises(M.V.SnapshotError): decode(value)

    def test_raw_bit_and_dimension_grammar_fails_closed(self):
        for bad in ('7ff0000000000000','7ff8000000000000','3FE5555555555555','0'*17,True,1):
            with self.subTest(bad=bad):
                value=fixture();value['coefficients'][0]=bad
                with self.assertRaises(M.V.SnapshotError): decode(value)
        for name, wrong in [('center',['0000000000000000']*11),('scale',['0000000000000000']*8),
                ('coefficients',['0000000000000000']*8)]:
            with self.subTest(name=name):
                value=fixture();value[name]=wrong
                with self.assertRaises(M.V.SnapshotError): decode(value)

    def test_external_model_and_recipe_pins_are_independent(self):
        value=fixture();body=M.V.canonical(value);recipe=sha256(M.V.canonical(value['recipe'])).hexdigest()
        for mp,rp in [('0'*64,recipe),(sha256(body).hexdigest(),'0'*64)]:
            with self.assertRaises(M.V.SnapshotError):
                M.decode_model(body,model_sha256=mp,recipe_sha256=rp)
        value['coefficients'][0]='3ff0000000000000'
        with self.assertRaises(M.V.SnapshotError):
            M.decode_model(M.V.canonical(value),model_sha256=sha256(body).hexdigest(),recipe_sha256=recipe)

    def test_canonical_json_bounds_duplicates_and_unknown_fields(self):
        value=fixture();body=M.V.canonical(value)
        malformed=[body.rstrip(b'\n'), b'{"schema":"wrong",'+body[1:], b'x'*(M.MAX_BYTES+1)]
        for raw in malformed:
            with self.assertRaises(M.V.SnapshotError):
                M.decode_model(raw,model_sha256=sha256(raw).hexdigest(),recipe_sha256='0'*64)
        value['extra']='field'
        with self.assertRaises(M.V.SnapshotError): decode(value)

    def test_bounded_ordinary_file_read_and_reparse_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(strict=True);path=root/'model.json';value=fixture();body=M.V.canonical(value)
            path.write_bytes(body)
            kwargs=dict(model_sha256=sha256(body).hexdigest(),recipe_sha256=sha256(M.V.canonical(value['recipe'])).hexdigest())
            self.assertEqual(M.read_model(path,**kwargs).coefficients[0],2/3)
            original=Path.lstat
            def reparse(item,*args,**kw):
                result=original(item,*args,**kw)
                if Path(item)==path:
                    class Marked:
                        st_file_attributes=0x400
                        def __getattr__(self,name): return getattr(result,name)
                    return Marked()
                return result
            with patch.object(Path,'lstat',reparse),self.assertRaises(M.V.SnapshotError):
                M.read_model(path,**kwargs)

    def test_prediction_fixture_rejects_layout_nonfinite_and_overflow(self):
        model=decode(fixture())
        for row in ([0.0]*11,[True]+[0.0]*7,[math.nan]+[0.0]*7,[10**1000]+[0.0]*7):
            with self.assertRaises(M.V.SnapshotError):M.predict_fixture(model,[row])
        huge=fixture(center=[M.bits(-1e308)]+['0000000000000000']*7)
        with self.assertRaises(M.V.SnapshotError):M.predict_fixture(decode(huge),[[1e308]+[0.0]*7])


if __name__=='__main__':
    unittest.main(verbosity=2)
