"""Independent scalar3.2 JSON prediction; preserved completed-run oracle."""
import json
import math
from bounded_bson import require, InvalidArtifact, f32

def xgb_predict(model,values,objective='reg:squarederror'):
    learner=model['learner']
    require(model['version']==[3,2,0] and learner['objective']['name']==objective,'imputer XGBoost schema')
    require(int(learner['learner_model_param']['num_feature'])==len(values),'imputer dimension')
    base=json.loads(learner['learner_model_param']['base_score'])
    require(isinstance(base,list) and len(base)==1,'imputer scalar base score')
    result=f32(float(base[0]))
    for tree in learner['gradient_booster']['model']['trees']:
        nodes=len(tree['left_children'])
        require(1<=nodes<=127 and all(len(tree[name])==nodes for name in
                ('right_children','split_type','split_indices','split_conditions','default_left')),
                'imputer tree shape/node budget')
        node=0
        # The immutable imputer recipe has maximum depth6: at most7 visited nodes.
        # A fixed hop bound also rejects cycles without allocating a visited set per point.
        for depth in range(7):
            require(type(node) is int and 0<=node<nodes,'imputer child outside tree')
            if tree['left_children'][node]==-1:
                require(tree['right_children'][node]==-1,'imputer leaf child mismatch')
                break
            require(depth<6,'imputer cyclic/deep tree exceeds fixed six splits')
            require(tree['split_type'][node]==0,'imputer categorical split unsupported')
            column=tree['split_indices'][node]
            require(0<=column<len(values),'imputer split feature')
            value=values[column]
            left=bool(tree['default_left'][node]) if math.isnan(value) else f32(value)<f32(tree['split_conditions'][node])
            node=tree['left_children'][node] if left else tree['right_children'][node]
        else:raise InvalidArtifact('imputer path did not terminate')
        result=f32(result+f32(tree['split_conditions'][node]))
    return result
