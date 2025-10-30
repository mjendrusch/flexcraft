import jax
import jax.numpy as jnp
from flexcraft.data.data import DesignData
from flexcraft.utils.rng import Keygen
from flexcraft.structure.af import AFResult, AFInput
import flexcraft.sequence.aa_codes as aas

def af_cycler(af_model, mpnn_model, confidence=None, fix_template=False):
    """AlphaFold2 recycling protocol.
    
    Args:
        af_model: AlphaFold2-like model.
        mpnn_model: ProteinMPNN-like model.
        confidence: confidence function to prescribe amount of noise added
            to the recycled sequence.
        fix_template: fix the template positions of all residues not in
            cycle_mask.
    
    Returns:
        cycle: function implementing a single cycling step.
    """
    if confidence is None:
        confidence = contact_confidence
    def cycle(af_params, key: Keygen, data: DesignData, cycle_mask=None):
        if cycle_mask is None:
            cycle_mask = jnp.ones_like(
                data["batch_index"], dtype=jnp.bool_)
        # run a single AlphaFold step with template, guess and structure module init
        template_aa = jnp.zeros_like(data["aa"])
        # should non-cycled parts be held fixed exactly, or be allowed to move?
        if fix_template:
            template_aa = jnp.where(cycle_mask, template_aa, data["aa"])
        af_input:  AFInput  = (
            AFInput.from_data(data)
            .add_template(data.update(
                aa=template_aa))
            .add_guess(data)
            .add_pos(data)
        )
        af_result: AFResult = af_model(af_params, key(), af_input)
        # compute confidence
        alpha = confidence(af_result)
        # get ProteinMPNN logits
        mpnn_input = af_result.to_data()
        mpnn_input = mpnn_input.update(aa=aas.translate(
            mpnn_input["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
        mpnn_input = mpnn_input.drop_aa(where=cycle_mask)
        mpnn_result = mpnn_model(key(), mpnn_input)
        logits = aas.translate_onehot(mpnn_result["logits"], aas.PMPNN_CODE, aas.AF2_CODE)
        logits -= logits.mean(axis=0)
        logits = logits[:, :20]
        # update_sequence
        start_aa = jax.nn.one_hot(af_input["aatype"], 20, axis=-1)
        noise = jax.random.gumbel(key(), logits.shape)
        update = alpha[:, None] * logits + (1 - alpha)[:, None] * noise
        new_aa = jnp.argmax(start_aa + update, axis=-1)
        # results
        prediction = af_result.to_data()
        update = data.update(
            aa=jnp.where(cycle_mask, new_aa, data["aa"]),
            atom_positions=prediction["atom_positions"],
            atom_mask=prediction["atom_mask"].at[:, 6:].set(False))
        return update, prediction
    return cycle

def contact_confidence(af_result: AFResult):
    p_contact = af_result.contact_probability()
    resi = af_result.inputs["residue_index"]
    distance = abs(resi[:, None] - resi[None, :])
    pair_mask = distance > 9
    p_contact = jnp.where(pair_mask, p_contact, 0.0)
    score = jnp.sort(p_contact, axis=1)[:, -2:].mean(axis=1)
    return score
