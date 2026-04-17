import jax
import jax.numpy as jnp
import optax
from flexcraft.hallucination.transform import logp_to_simplex, project_to_simplex

def jit_loss_update(loss):
    @jax.jit
    def loss_update(x, key=None, context=None, params=None):
        (value, aux), grad = jax.value_and_grad(loss, argnums=0, has_aux=True)(
            x, key=key, context=context, params=params)
        return (value, aux), grad
    return loss_update

def transform_opt_step(loss_update, key=None, params=None, lr=0.1,
                       scale=1.0, grad_transform: optax.GradientTransformation=None,
                       input_transform=None, verbose=False, logdir="./"):
    def _step(state, transform_state=None, context=None):
        x, x_prev = state
        x_prev = x
        x_in = jax.nn.softmax(x, axis=-1)
        (loss, aux), grad = loss_update(
            jnp.array(x_in), key=jnp.array(key()), context=context, params=params)
        if grad_transform is not None:
            if transform_state is None:
                transform_state = grad_transform.init(x)
            grad, transform_state = grad_transform.update(grad, transform_state, x)
        x = x - lr * grad
        if verbose:
            _report_aux(aux)
        return (x, x_prev), transform_state, (loss, aux)
    return _step

def simplex_apgm_step(loss_update, key=None, params=None, lr=0.1, momentum=0.0,
                      scale=1.0, opt_logits=False, softmax_grad=False,
                      grad_transform: optax.GradientTransformation=None,
                      param_transform=None, verbose=False, logdir="./"):
    def _step(state, transform_state=None, context=None):
        x, x_prev = state
        x_m = x + momentum * (x - x_prev)
        x_prev = x
        if opt_logits:
            x_in = jax.nn.softmax(x_m)
        else:
            x_in = x_m
        (loss, aux), grad = loss_update(
            jnp.array(x_in), key=jnp.array(key()), context=context, params=params)
        # optionally _do not_ use the straight-through gradient
        # for the softmax on the logits and compute the proper
        # gradient by composing with the softmax jvp
        if opt_logits and softmax_grad:
            grad = jax.jvp(jax.nn.softmax, [x_m], [grad])
        if grad_transform is not None:
            if transform_state is None:
                transform_state = grad_transform.init(x)
            grad, transform_state = grad_transform.update(grad, transform_state, x)
        x = x_m - lr * grad
        if opt_logits:
            x = scale * x
        else:
            x = project_to_simplex(scale * x)
        if param_transform is not None:
            x = param_transform(x)
        if verbose:
            _report_aux(aux)
        return (x, x_prev), transform_state, (loss, aux)
    return _step

def _report_aux(aux):
    res = []
    for key, value in aux.items():
        if key == "result":
            continue
        res.append(f"{key} = {value:.2f}")
    print(*res)

def simplex_agpm_state(x):
    return (x, x)

def transform_opt_state(x):
    return (x, x)

def optimize(step, init_state, context=None,
             num_steps=100, return_last=False,
             early_stop=None, report=None):
    if context is None:
        context = dict()
    def _run_opt(x):
        state = init_state(x)
        tf_state = None
        best = x
        best_aux = None
        prev_x = x
        best_val = jnp.inf
        for i in range(num_steps):
            context["step"] = i
            state, tf_state, (loss, aux) = step(
                state, tf_state, context=context)
            if report is not None:
                report(i, loss, aux)
            if best_aux is None:
                best_aux = aux
            if loss < best_val:
                best = state[0]
                best_val = loss
                best_aux = aux
            if early_stop is not None:
                if early_stop(best_val, best_aux):
                    break
        if return_last:
            return state[0], loss, aux
        return best, best_val, best_aux
    return _run_opt

def simplex_agpm(x, loss_update, key=None, context=None, params=None, num_steps=100, lr=0.1, momentum=0.0,
                 scale=1.0, opt_logits=False, softmax_grad=False,
                 grad_transform=None, param_transform=None, verbose=False, logdir="./",
                 return_last=False, early_stop=None, report=None):
    step = simplex_apgm_step(
        loss_update, key=key, params=params, lr=lr, momentum=momentum,
        scale=scale, opt_logits=opt_logits, softmax_grad=softmax_grad,
        grad_transform=grad_transform, param_transform=param_transform,
        verbose=verbose, logdir=logdir)
    opt = optimize(step, simplex_agpm_state, context=context,
                   num_steps=num_steps, return_last=return_last,
                   early_stop=early_stop, report=report)
    best, best_val, best_aux = opt(x)
    if opt_logits:
        best = jax.nn.softmax(best)
    return best, best_val, best_aux

def transform_opt(x, loss_update, key=None, context=None, params=None, num_steps=100, lr=0.1,
                  scale=1.0, grad_transform: optax.GradientTransformation=None,
                  input_transform=None, verbose=False, logdir="./",
                  return_last=False, early_stop=None):
    state = transform_opt_state(x)
    step = transform_opt_step(
        loss_update, key=key, params=params, lr=lr,
        scale=scale, grad_transform=grad_transform,
        input_transform=input_transform,
        verbose=verbose, logdir=logdir)
    opt = optimize(step, transform_opt_state, context=context,
                   num_steps=num_steps, return_last=return_last,
                   early_stop=early_stop)
    return opt(x)
