"""Pre-code literal scalar XGBoost expectations; no author or ML imports."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
from bounded_bson import InvalidArtifact
from xgboost_scalar import xgb_predict,official_model_version

def leaf(value):
    return {'left_children':[-1],'right_children':[-1],'split_type':[0],
            'split_indices':[0],'split_conditions':[value],'default_left':[False]}

def model():
    tree={'left_children':[1,-1,-1],'right_children':[2,-1,-1],'split_type':[0,0,0],
          'split_indices':[0,0,0],'split_conditions':[1.,-1.5,2.75],'default_left':[True,False,False]}
    return {'version':[3,2,0],'learner':{'objective':{'name':'reg:squarederror'},
        'learner_model_param':{'num_feature':'1','base_score':'[0.25]'},
        'gradient_booster':{'model':{'trees':[tree]}}}}

class LiteralScalarReplay(unittest.TestCase):
    def test_version_components_are_integers(self):
        from prepare_artifact_audit import validate_imputer_model
        self.assertTrue(official_model_version([3,2,0]))
        for version in ([3,2,False],[3.,2,0],[3,2.,0],[3,2,0.],(3,2,0),'3.2.0'):
            with self.subTest(version=version):
                self.assertFalse(official_model_version(version))
                value=model();value['version']=version
                with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])
                with self.assertRaisesRegex(ValueError,'integer-version'):
                    validate_imputer_model({'version':version,'learner':None},[1])

    def test_literal_threshold_and_default(self):
        value=model()
        self.assertEqual(xgb_predict(value,[0.]),-1.25)
        self.assertEqual(xgb_predict(value,[1.]),3.)
        self.assertEqual(xgb_predict(value,[2.]),3.)
        self.assertEqual(xgb_predict(value,[math.nextafter(1.,0.)]),3.)
        self.assertEqual(xgb_predict(value,[1.-2**-24]),-1.25)
        self.assertEqual(xgb_predict(value,[math.nan]),-1.25)
        value['learner']['gradient_booster']['model']['trees'][0]['default_left'][0]=False
        self.assertEqual(xgb_predict(value,[math.nan]),3.)

    def test_per_tree_float32_accumulation(self):
        value=model()
        value['learner']['learner_model_param']['base_score']='[16777216]'
        value['learner']['gradient_booster']['model']['trees']=[leaf(1.),leaf(-16777216.)]
        self.assertEqual(xgb_predict(value,[0.]),0.)

    def test_fixed_version_objective_and_shape(self):
        for field,bad in (('version',[3,1,0]),('objective','reg:absoluteerror')):
            value=model()
            if field=='version':value['version']=bad
            else:value['learner']['objective']['name']=bad
            with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])
        with self.assertRaises(InvalidArtifact):xgb_predict(model(),[0.,1.])
        value=model()
        value['learner']['gradient_booster']['model']['trees'][0]['right_children'].pop()
        with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])
        value=model()
        value['learner']['gradient_booster']['model']['trees'][0]['split_type'][0]=1
        with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])

    def test_cyclic_children_and_index(self):
        value=model()
        value['learner']['gradient_booster']['model']['trees'][0]['left_children'][0]=0
        with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])
        value=model()
        value['learner']['gradient_booster']['model']['trees'][0]['split_indices'][0]=1
        with self.assertRaises(InvalidArtifact):xgb_predict(value,[0.])

if __name__=='__main__':unittest.main()
