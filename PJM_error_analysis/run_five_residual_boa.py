#!/usr/bin/env python3
"""Five existing online experts -> five independent residual LSTMs -> BOA.
Run from anywhere: python ~/run_five_residual_boa.py --pull --device cpu
Default: PJM/split2_2018. Reuses the repository's residual training and BOA implementations.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def find_repo(explicit):
    def valid(p):
        return (p / '.git').exists() and (p / 'adaptive-weighted-LSTM-Attention').is_dir()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not valid(p):
            raise SystemExit(f'저장소 경로를 확인하세요: {p}')
        return p
    for root in (Path.cwd(), Path(__file__).resolve().parent):
        for p in (root, *root.parents):
            if valid(p):
                return p
    found = []
    excluded = {'.git', '.cache', '.Trash', '.venv', 'venv', 'node_modules', 'Library', 'Downloads', 'anaconda3', 'miniconda3', 'miniforge3', 'envs'}
    home = Path.home()
    for root, dirs, _ in os.walk(home):
        p = Path(root)
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded]
        if valid(p):
            found.append(p)
            dirs[:] = []
        if len(p.relative_to(home).parts) >= 5:
            dirs[:] = []
    if len(found) != 1:
        raise SystemExit('저장소를 하나로 특정하지 못했습니다. --repo "저장소 절대경로" 를 붙여주세요.\n' + '\n'.join(map(str, found)))
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo')
    p.add_argument('--pull', action='store_true', help='git pull --ff-only before execution')
    p.add_argument('--market', default='PJM', choices=['PJM', 'DE'])
    p.add_argument('--split', default='split2_2018', choices=['split2_2018', 'split1_2017'])
    p.add_argument('--input-dir', type=Path, help='directory containing predictions_val.csv and predictions_test.csv')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--seeds', type=int, default=5, help='seed averages within each of the five experts')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--losses', nargs='+', default=['mse', 'l1'], choices=['mse', 'l1'])
    args = p.parse_args()
    if min(args.seeds, args.epochs, args.patience, args.batch, args.threads, args.n_boot) < 1:
        p.error('학습/평가 횟수는 1 이상이어야 합니다.')
    repo = find_repo(args.repo)
    if args.pull:
        subprocess.run(['git', 'pull', '--ff-only'], cwd=repo, check=True)
    for rel in ['ensemble/boa/boa_lstm.py', 'ensemble/boosting/lear_lstm_boost.py']:
        if not (repo / rel).is_file():
            raise SystemExit(f'{rel}이 없습니다. 저장소를 git pull --ff-only 로 갱신하세요.')
    sys.path[:0] = [str(repo / 'ensemble/boa'), str(repo / 'ensemble/boosting')]
    try:
        import numpy as np
        import pandas as pd
        import torch
        import boa_lstm as online
        import lear_lstm_boost as residual
    except ImportError as e:
        raise SystemExit(f'{e}\n현재 Python 환경에 설치: python -m pip install torch numpy pandas scipy scikit-learn') from e
    device = residual.baseline.setup(args.device, args.threads)
    candidates = [repo / 'adaptive-weighted-LSTM-Attention/results' / online.tree.EXPORT_TAG[args.market] / args.split / 'export']
    candidates += sorted(repo.glob(f'experiment_archive/*/project/adaptive-weighted-LSTM-Attention/results/{online.tree.EXPORT_TAG[args.market]}/{args.split}/export'))
    inputs = args.input_dir.expanduser().resolve() if args.input_dir else next((q for q in candidates if all((q / f'predictions_{k}.csv').is_file() for k in ['val', 'test'])), None)
    if inputs is None:
        raise SystemExit('A/LEAR 검증·테스트 예측이 없습니다. --input-dir 로 export 폴더를 지정하세요.')
    out = args.output_dir.expanduser().resolve() if args.output_dir else repo / 'ensemble/boa/results' / args.market / args.split / 'five_individual_residual'
    if (out / 'selection.json').exists():
        raise SystemExit(f'완료된 결과가 있습니다: {out}\n다시 실행하려면 --output-dir 로 새 폴더를 지정하세요.')
    out.mkdir(parents=True, exist_ok=True)
    dates, parts, d = online.daily_table(inputs, ['actual', 'A', 'LEAR'])
    Y, A, L = d['actual'], d['A'], d['LEAR']
    assert all(np.isfinite(x).all() for x in [Y, A, L])
    val, test = np.flatnonzero(parts == 'val'), np.flatnonzero(parts == 'test')
    assert len(val) == 364 and len(test) == 364 and val[-1] < test[0]
    assert dates.equals(pd.date_range(dates[0], periods=len(dates), freq='D'))
    # All boundaries are fixed before inspecting test performance.
    warm = val[:60]
    base_select = val[60:120]
    train = val[60:182]
    res_select = val[182:243]
    boa_select = val[243:304]
    hold = val[304:]
    scale = max(float(Y[warm].std()), 1e-6)
    m = online.data.load(args.split, online.tree.CSV[args.market])
    ids = m.days.get_indexer(dates)
    assert (ids >= 0).all() and np.allclose(m.truth[ids], Y, atol=.002, rtol=0)
    assert np.array_equal(ids[val], np.arange(m.ntrain, m.ntrain + m.nval))
    raw_table = pd.read_csv(online.tree.CSV[args.market], index_col=0)
    raw_table.index = pd.to_datetime(raw_table.index)
    positions = raw_table.index.get_indexer(dates)
    print(f'Repo: {repo}\nInput: {inputs}\nOutput: {out}', flush=True)
    print('검증 구간: 0:60 준비 / 60:120 기존 보정 선택 / 60:182 잔차 학습 / 182:243 잔차 선택 / 243:304 BOA 선택 / 304:364 확인', flush=True)
    print('기존 보정 선택 구간은 잔차 학습 구간 안에 포함됩니다. 잔차 선택·BOA·확인·테스트 구간은 이후 날짜입니다.', flush=True)
    aggs, S = online.aggregate_forecasts(raw_table, positions, dates, out)
    base_preds, base_configs, _ = online.online_members(Y, A, L, warm, base_select, scale, aggs, S)
    mean, std = m.scaling(m.ntrain)
    pm, ps = float(mean[0]), float(std[0])
    history, forecast, weekday, _ = m.tensors(mean, std)
    shared = tuple(t[ids] for t in (history, forecast, weekday))
    tr = torch.as_tensor(train)
    va = torch.as_tensor(res_select)
    all_idx = torch.arange(len(Y))
    names = ['TrainingRecent', 'MeanError28_half', 'LevelShape_RMSE', 'Reconcile_RMSE', 'BiasMix_RMSE']
    corrected, configs, histories = [], {}, []
    for name in names:
        base = base_preds[name]
        # Same ResidualLSTM architecture. Replace the LEAR/gap inputs by this expert's forecast and A-base gap.
        extra = np.stack([(base - pm) / ps, (A - base) / ps], axis=-1).astype('float32')
        target = ((Y - base) / ps).astype('float32')
        T = (*shared, torch.from_numpy(extra), torch.from_numpy(target))
        options = []
        for loss in args.losses:
            seed_predictions = []
            for seed in range(args.seeds):
                model, info = residual.train_one(T, tr, va, seed, loss, device, args)
                pred = residual.predict(model, T, all_idx, device) * ps
                seed_predictions.append(pred)
                histories.append({'expert': name, 'loss': loss, **info})
                checkpoint = out / 'checkpoints' / name
                checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'mean': mean.tolist(), 'std': std.tolist(), 'expert': name, 'loss': loss, 'seed': seed},
                           checkpoint / f'{loss}_seed{seed}.pt')
                print(f'{name} {loss} seed={seed} epoch={info["best_epoch"]}', flush=True)
            rhat = np.mean(seed_predictions, axis=0)
            c = min(online.tree.SHRINK, key=lambda c: np.mean((Y[res_select] - base[res_select] - c*rhat[res_select])**2))
            pr = base + c * rhat
            options.append((online.score(Y[res_select], pr[res_select])['RMSE'], loss, float(c), pr))
        _, loss, c, best = min(options, key=lambda x: x[0])
        corrected.append(best)
        configs[name] = {'loss': loss, 'c': c, 'seed_count': args.seeds}
        print(f'Selected {name}: {loss}, c={c:g}', flush=True)
    X0 = np.stack([base_preds[n] for n in names], axis=-1)
    X1 = np.stack(corrected, axis=-1)
    predictions = {'LEAR': L, **{n: base_preds[n] for n in names},
                   **{f'Residual:{n}': v for n, v in zip(names, corrected)},
                   'Mean5': X0.mean(-1), 'Mean5Residual': X1.mean(-1)}
    weights, trials, selections = {}, [], {}
    for tag, experts in [('BOA5_control', X0), ('BOA5_each_residual', X1)]:
        opts = []
        for loss, groups in itertools.product(['MAE', 'MSE'], [1, 4]):
            pr, w = online.boa(Y, experts, loss, groups, int(boa_select[0]), scale)
            s = online.score(Y[boa_select], pr[boa_select])
            cfg = {'loss': loss, 'hour_groups': groups}
            trials.append({'model': tag, **cfg, **s})
            opts.append((s['RMSE'], cfg, pr, w))
        _, selections[tag], predictions[tag], weights[tag] = min(opts, key=lambda x: x[0])
    # Nothing below this line chooses a model using test labels.
    rows = []
    for name, pr in predictions.items():
        rows.append({'model': name, **{f'{part}_{k}': v for part, idx in [('holdout', hold), ('test', test)] for k, v in online.score(Y[idx], pr[idx]).items()}})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'scores.csv', index=False)
    boots = []
    for ref in ['BOA5_control', 'Mean5Residual']:
        for loss, fn in online.tree.LOSSES.items():
            result = online.metrics.block_bootstrap(fn(predictions['BOA5_each_residual'][test] - Y[test]), fn(predictions[ref][test] - Y[test]), n_boot=args.n_boot)
            boots.append({'model': 'BOA5_each_residual', 'vs': ref, 'loss': loss, **result})
    pd.DataFrame(boots).to_csv(out / 'bootstrap.csv', index=False)
    pd.DataFrame(histories).to_csv(out / 'training.csv', index=False)
    pd.DataFrame(trials).to_csv(out / 'boa_selection_trials.csv', index=False)
    for part, idx in [('holdout', hold), ('test', test)]:
        stamps = (dates.values[idx, None] + np.arange(24) * np.timedelta64(1, 'h')).ravel()
        pd.DataFrame({'Date': stamps, 'actual': Y[idx].ravel(), **{k: v[idx].ravel() for k, v in predictions.items()}}).to_csv(out / f'predictions_{part}.csv', index=False)
    for tag, w in weights.items():
        np.savez_compressed(out / f'{tag}_weights.npz', dates=dates.values, weights=w, experts=np.array(names))
    np.savez_compressed(out / 'all_expert_predictions.npz', dates=dates.values, parts=parts, actual=Y, original=X0, residual_corrected=X1, experts=np.array(names))
    metadata = {'market': args.market, 'split': args.split, 'input': str(inputs), 'experts': names,
                'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'input_sha256': {part: hashlib.sha256((inputs / f'predictions_{part}.csv').read_bytes()).hexdigest() for part in ['val', 'test']},
                'base_configs': base_configs, 'residual_configs': configs, 'boa_configs': selections,
                'protocol': {'warm': [0, 60], 'base_select': [60, 120], 'residual_train': [60, 182], 'residual_select': [182, 243], 'boa_select': [243, 304], 'holdout': [304, 364]},
                'limitations': ['Residual training uses 122 validation days; historical OOF training is not included.', 'Base configurations are selected inside residual training, so residual training inputs are not strict OOF.', 'The original five use RMSE-selected correction configurations; loss and shrinkage are selected by residual-selection RMSE.', 'This control uses the same revised time boundaries; it is not a reproduction of the previously reported BOA5.'],
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (out / 'selection.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(scores[['model', 'holdout_MAE', 'holdout_RMSE', 'test_MAE', 'test_RMSE']].round(4).to_string(index=False), flush=True)
    print(f'완료: {out}', flush=True)


if __name__ == '__main__':
    main()
