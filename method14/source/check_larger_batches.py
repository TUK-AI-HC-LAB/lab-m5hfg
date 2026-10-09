"""Compare local batch throughput; separate run, fixed epochs, finite gradients."""
import os
os.environ['IGD_RUN_TAG']='mvtec_largebatch_seed42_20261007'
os.environ['IGD_ACCELERATED']='1'
from pathlib import Path
import json
import textwrap
source=(Path(__file__).parent/'check_local_acceleration.py').read_text()
source=source.replace('mvtec_accelerated_seed42_20261007','mvtec_largebatch_seed42_20261007')
prefix=source[:source.index('for accelerated in [False,True]:')]
body=source[source.index('    torch.backends.cuda.matmul'):source.index("(OUT/'acceleration_preflight.json')")]
body=body.replace('torch.randn(450,','torch.randn(batch_size*225,').replace('torch.rand(450,','torch.rand(batch_size*225,')
body=body.replace('    times=[];losses=[]','    torch.cuda.reset_peak_memory_stats()\n    times=[];losses=[]')
body=body.replace('results.append(dict(accelerated=accelerated,','results.append(dict(batch_size=batch_size,peak_allocated_bytes=torch.cuda.max_memory_allocated(),images_per_second=batch_size/(sum(times[1:])/3),accelerated=accelerated,')
exec(prefix)
for batch_size in [2,4,8]:
    accelerated=True
    try:
        exec(textwrap.dedent(body))
    except torch.OutOfMemoryError:
        results.append(dict(batch_size=batch_size,status='out_of_memory'))
        break
    (OUT/'batch_preflight.json').write_text(json.dumps(dict(results=results,conditions='representative full local optimization step; fixed random input; warmup excluded; finite gradients checked'),indent=2))
(OUT/'batch_preflight.json').write_text(json.dumps(dict(results=results,conditions='representative full local optimization step; fixed random input; warmup excluded; finite gradients checked'),indent=2))
print(json.dumps(results))
