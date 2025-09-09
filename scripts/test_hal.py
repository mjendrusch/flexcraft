import os
import time
import jax
from colabdesign.af.alphafold.model.config import model_config
from colabdesign.af.alphafold.model.data import get_model_haiku_params
from salad.aflib.common.protein import to_pdb, from_prediction

from flexcraft.utils.options import parse_options
from flexcraft.utils.rng import Keygen
from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer
from flexcraft.sequence.sample import *
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.structure.af import soft_sequence, forbid_sequence, AFInput, AFResult, make_af2, make_predict
# from colabdesign.af.alphafold.common.protein import from_prediction, to_pdb
opt = parse_options(
    "predict structures with AlphaFold",
    param_path="params/",
    pmpnn_path="../prosesame/v_48_030.pkl",
    model_name="model_1_ptm",
    out_path="out",
    center="False",
    length=50,
    repeat=1,
    temperature=0.1,
    samples=10,
    seed=42
)
# set up alphafold model
params = get_model_haiku_params(
    model_name=opt.model_name,
    data_dir=opt.param_path, fuse=True)
config = model_config(opt.model_name)
config.model.global_config.use_dgram = False
config.model.global_config.use_remat = True
af2 = make_predict(make_af2(config), num_recycle=2)
key = Keygen(opt.seed)
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_path, eps=0.05))

# fitness function for hallucination
def fitness(params, key, sequence, soft=0.0, hard=0.0, T=1.0):
    sequence = soft_sequence(sequence, soft=soft, hard=hard, temperature=T)
    sequence = forbid_sequence(sequence, value=0)
    repeat = jnp.concatenate(opt.repeat * [sequence], axis=0)
    data = AFInput.from_sequence(repeat)
    result = af2(params, key, data)
    resi = result.inputs["residue_index"]
    close = abs(resi[:, None] - resi[None, :]) < 10
    contact = result.contact_entropy(contact_distance=14.0)
    contact = jnp.where(close, 1e6, contact)
    contact_sum = -contact.sort(axis=1)[:, :2].mean()
    metrics = dict(result=result,
                   pae=result.pae.mean(),
                   plddt=result.plddt.mean(),
                   pcontact=-contact_sum)
    value = contact_sum + metrics["plddt"] + (1 - metrics["pae"])
    return value, metrics
def gradient_step(params, key, sequence, soft=0.0, hard=0.0, T=1.0):
    (value, aux), grad = jax.value_and_grad(fitness, argnums=(2,), has_aux=True)(
        params, key, sequence, soft, hard, T)
    lr = (1 - soft) + (soft * T)
    grad = grad[0]
    grad = forbid_sequence(grad, value=0.0)
    grad /= jnp.sqrt(jnp.maximum((grad ** 2).sum(), 1e-3))
    new_sequence = sequence + lr * grad
    return forbid_sequence(new_sequence, value=0), value, aux
gradient_step = jax.jit(gradient_step)

os.makedirs(opt.out_path, exist_ok=True)
sequence = jax.nn.softmax(
    forbid_sequence(jax.random.gumbel(key(), (opt.length, 20)), -1e9))
best = -1e6
result = None
for idx in range(100):
    t = 1 - (idx + 1) / 100
    soft = 0.0#1 - t
    start = time.time()
    sequence, value, aux = gradient_step(params, key(), sequence, soft=soft)
    print(f"Scores at step {idx} in {time.time() - start:.1f} s", 
          f"fitness: {value:.2f}", f"pae: {aux['pae']:.2f}",
          f"plddt: {aux['plddt']:.2f}", f"contact: {aux['pcontact']:.2f}")
    if value > best:
        result: AFResult = aux["result"]
        best = value
        result.save_pdb(f"{opt.out_path}/step_{idx}.pdb")
