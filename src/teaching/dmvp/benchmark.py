import re
import subprocess

import pandas as pd


versions = ['base', 'omp-target.v0', 'cuda.v0', 'cuda.v1', 'cuda.v2', 'cuda.v3', 'cuda.v4', 'cuda.v5', 'cuda.v6', 'cuda.cublas', 'omp-target.v7', 'omp-target.v8', 'omp-target.v9']
columns = ['nx', *versions]


def eval_gpu():
    out = subprocess.check_output(['nvidia-smi', '-L'])
    out = out.decode('utf-8').strip()
    gpu = re.findall(r'GPU 0: (.*) \(UUID: GPU', out)[0]
    gpu_for_filename = gpu.replace(' ', '-').replace('NVIDIA-', '').replace('GeForce-', '')
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
    out = out.decode('utf-8').strip()
    gpu_cc = float(out)

    print(f'Running on GPU {gpu} ({gpu_for_filename}), compute capability {gpu_cc}')

    return gpu_for_filename


def measure(tpe, nx, version):
    # execute
    out = subprocess.check_output([f'./build/dmvp-{version}', *[f'{p}' for p in [tpe, nx]]])
    out = out.decode("utf-8")

    # parse
    bandwidth = float(re.findall(r'bandwidth: *(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+) GB/s', out)[0])

    return bandwidth


def measure_versions(tpe, nx):
    print(f'Measuring {tpe} with nx={nx}...')

    measured = {v : measure(tpe, nx, v) for v in versions}

    return {
        'nx': nx,
        **measured
    }


def main():
    gpu = eval_gpu()

    nx_to_scan = []
    samples_per_interval = 8

    min_nx_exp = 5
    max_nx_exp = 16
    for nx_exp in range(min_nx_exp, max_nx_exp):
        for i in range(samples_per_interval):
            to_append = int(round(2 ** (nx_exp + i / samples_per_interval) / 32)) * 32
            if to_append not in nx_to_scan:
                nx_to_scan.append(to_append)

    nx_to_scan.append(2 ** max_nx_exp)

    for tpe in ['float', 'double']:
        results = [measure_versions(tpe, nx) for nx in nx_to_scan]

        df = pd.DataFrame(results, columns=columns)

        df.to_csv(f'./measured/{gpu}-{tpe}.csv', index=True, index_label="index")
        df.to_excel(f'./measured/{gpu}-{tpe}.xlsx', index=True, index_label="index")


if __name__ == '__main__':
    main()
