"""Offline, timestamp-validated inference for the packaged three-station case."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.lstm_predictor import LSTMPredictor


def validate_history(frame,config):
    cols=['timestamp',*config['feature_columns']]
    missing=set(cols)-set(frame.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(frame)!=config['history_hours']:raise ValueError('Exactly 24 hourly rows are required')
    parsed=pd.to_datetime(frame.timestamp)
    if parsed.dt.tz is None:raise ValueError('Timestamp must contain an explicit timezone offset')
    parsed=parsed.dt.tz_convert(config['timezone'])
    if not (parsed.diff().iloc[1:]==pd.Timedelta(hours=1)).all():
        raise ValueError('Timestamps must be strictly ascending with 1-hour intervals')
    if not parsed.eq(parsed.dt.floor('h')).all():raise ValueError('Timestamps must be aligned to whole hours')
    values=frame[config['feature_columns']].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Input contains missing, nonfinite, or negative concentrations')
    return frame[config['feature_columns']].copy(),parsed.iloc[-1]


def forecast(bundle,input_csv,method=None):
    bundle=Path(bundle)
    config=json.loads((bundle/'inference_config.json').read_text())
    frame,origin=validate_history(pd.read_csv(input_csv),config)
    method=method or config['selected_model']
    if method=='persistence':prediction=np.repeat(float(frame.pm25.iloc[-1]),6)
    elif method=='previous_day':prediction=frame.pm25.iloc[:6].to_numpy()
    elif method in ['single_station','multi_station']:
        predictions=[]
        for seed in config['seeds']:
            model=LSTMPredictor.load(str(bundle/f'models/{method}_{seed}.pt'),device=torch.device('cpu'))
            predictions.append(model.predict(frame))
        prediction=np.mean(predictions,axis=0)
    else:raise ValueError(f'Unsupported model: {method}')
    return {'target_station':config['target_station'],'forecast_origin':origin.isoformat(),
            'model':method,'unit':'ug/m3','scope':config['scope'],
            'forecast':[{'timestamp':(origin+pd.Timedelta(hours=h+1)).isoformat(),
                         'horizon_h':h+1,'pm25':float(value)} for h,value in enumerate(prediction)],
            'limitations':'Concentration forecast only; no verified causal class, source location, or emission contribution'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',default='data/beijing_neighborhood_2019')
    parser.add_argument('--input',required=True)
    parser.add_argument('--model',choices=['single_station','multi_station','persistence','previous_day'])
    parser.add_argument('--output')
    args=parser.parse_args()
    result=json.dumps(forecast(args.bundle,args.input,args.model),ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(result)
    print(result)


if __name__=='__main__':main()
