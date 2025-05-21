# adapted from ColabFold

import joblib

import functools
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp

Gelu = functools.partial(jax.nn.gelu, approximate=False)


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape([neighbor_idx[None].shape[0], -1])
    neighbors_flat = jnp.tile(
        jnp.expand_dims(neighbors_flat, -1), [1, 1, nodes[None].shape[2]]
    )
    # Gather and re-pack
    neighbor_features = jnp.take_along_axis(nodes[None], neighbors_flat, 1)
    neighbor_features = neighbor_features.reshape(
        list(neighbor_idx[None].shape[:3]) + [-1]
    )
    return neighbor_features[0]


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)[None]
    h_nn = jnp.concatenate([h_neighbors[None], h_nodes], -1)
    return h_nn[0]


def get_sep_mask(order, filled):
    # all unfilled positions sit at max index + 1
    order = jnp.where(filled, order, order.max() + 1)
    # allow to attend on all earlier positions
    mask = order[:, None] > order[None, :]
    # results in a mask like
    # fill : f f f u u
    # order: 0 1 2 3 3
    # mask : 0 0 0 0 0
    #        1 0 0 0 0
    #        1 1 0 0 0
    #        1 1 1 0 0 <- all unfilled positions only
    #        1 1 1 0 0    attend to filled positions
    return mask


def drop(x, rate=0.1):
    return hk.dropout(hk.next_rng_key(), rate, x)


class EncLayer(hk.Module):
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None
    ):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = dropout

        self.norm1 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm1"
        )
        self.norm2 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm2"
        )
        self.norm3 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm3"
        )

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + "_W1")
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + "_W2")
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + "_W3")
        self.W11 = hk.Linear(num_hidden, with_bias=True, name=name + "_W11")
        self.W12 = hk.Linear(num_hidden, with_bias=True, name=name + "_W12")
        self.W13 = hk.Linear(num_hidden, with_bias=True, name=name + "_W13")
        self.act = Gelu
        self.dense = PositionWiseFeedForward(
            num_hidden, num_hidden * 4, name=name + "_dense"
        )

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, tie_info=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        # tie neighbour features across tied positions
        if tie_info is not None:
            dh_weighted = dh * tie_info["tie_weights"]
            dh = jnp.zeros_like(dh).at[tie_info["tie_index"]].add(dh_weighted)

        h_V = self.norm1(h_V + drop(dh, rate=self.dropout))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + drop(dh, rate=self.dropout))
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + drop(h_message, rate=self.dropout))
        return h_V, h_E


class DecLayer(hk.Module):
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None
    ):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = dropout
        self.norm1 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm1"
        )
        self.norm2 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm2"
        )

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + "_W1")
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + "_W2")
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + "_W3")
        self.act = Gelu
        self.dense = PositionWiseFeedForward(
            num_hidden, num_hidden * 4, name=name + "_dense"
        )

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None, tie_info=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_E.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        # tie neighbour features across tied positions
        if tie_info is not None:
            dh_weighted = dh * tie_info["tie_weights"]
            dh = jnp.zeros_like(dh).at[tie_info["tie_index"]].add(dh_weighted)
        
        h_V = self.norm1(h_V + drop(dh, rate=self.dropout))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + drop(dh, rate=self.dropout))

        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(hk.Module):
    def __init__(self, num_hidden, num_ff, name=None):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = hk.Linear(num_ff, with_bias=True, name=name + "_W_in")
        self.W_out = hk.Linear(num_hidden, with_bias=True, name=name + "_W_out")
        self.act = Gelu

    def __call__(self, h_V):
        h = self.act(self.W_in(h_V), approximate=False)
        h = self.W_out(h)
        return h


class PositionalEncodings(hk.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = hk.Linear(num_embeddings, name="embedding_linear")

    def __call__(self, offset, mask):
        d = jnp.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot)
        return E


class ProteinFeatures(hk.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        num_chain_embeddings=16,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = hk.Linear(
            edge_features, with_bias=False, name="edge_embedding"
        )
        self.norm_edges = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="norm_edges"
        )

    def _get_edge_idx(self, X, mask, batch, eps=1e-6):
        """get edge index
        input: mask.shape = (...,L), X.shape = (...,L,3)
        return: (...,L,k)
        """
        same_batch = batch[:, None] == batch[None, :]
        mask_2D = mask[..., None, :] * mask[..., :, None]
        mask_2D *= same_batch
        dX = X[..., None, :, :] - X[..., :, None, :]
        D = jnp.sqrt(jnp.square(dX).sum(-1) + eps)
        D_masked = jnp.where(mask_2D, D, D.max(-1, keepdims=True))
        k = min(self.top_k, X.shape[-2])
        return jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]

    def _rbf(self, D):
        """radial basis function (RBF)
        input: (...,L,k)
        output: (...,L,k,?)
        """
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        return jnp.exp(-(((D[..., None] - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A, B, E_idx):
        D = jnp.sqrt(jnp.square(A[..., :, None, :] - B[..., None, :, :]).sum(-1) + 1e-6)
        D_neighbors = jnp.take_along_axis(D, E_idx, 1)
        return self._rbf(D_neighbors)

    def __call__(self, data):
        if self.augment_eps > 0:
            X = data["atom_positions"] + self.augment_eps * jax.random.normal(
                hk.next_rng_key(), data["atom_positions"].shape
            )
        else:
            X = data["atom_positions"]

        # autoconvert atom14 and atom24 to atom4
        if X.shape[1] in (14, 24):
            X = X[:, :4]

        # get atoms
        # N,Ca,C,O,Cb
        Y = X.swapaxes(0, 1)  # (length, atoms, 3) -> (atoms, length, 3)
        if Y.shape[0] == 4:
            # add Cb
            b, c = (Y[1] - Y[0]), (Y[2] - Y[1])
            Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
            Y = jnp.concatenate([Y, Cb[None]], 0)

        # gather edge features
        # get edge indices (based on ca-ca distances)
        neighbours = self._get_edge_idx(Y[1], data["mask"], data["batch_index"])

        # rbf encode distances between atoms
        edges = jnp.array(
            [
                [1, 1],
                [0, 0],
                [2, 2],
                [3, 3],
                [4, 4],
                [1, 0],
                [1, 2],
                [1, 3],
                [1, 4],
                [0, 2],
                [0, 3],
                [0, 4],
                [4, 2],
                [4, 3],
                [3, 2],
                [0, 1],
                [2, 1],
                [3, 1],
                [4, 1],
                [2, 0],
                [3, 0],
                [4, 0],
                [2, 4],
                [3, 4],
                [2, 3],
            ]
        )
        RBF_all = jax.vmap(lambda x: self._get_rbf(Y[x[0]], Y[x[1]], neighbours))(edges)
        RBF_all = RBF_all.transpose((1, 2, 0, 3))
        RBF_all = RBF_all.reshape(RBF_all.shape[:-2] + (-1,))

        ##########################
        # position embedding
        ##########################
        # residue index offset
        if "offset" not in data:
            data["offset"] = data["residue_index"][:, None] - data["residue_index"][None, :]
        offset = jnp.take_along_axis(data["offset"], neighbours, 1)

        # chain index offset
        E_chains = (data["chain_index"][:, None] == data["chain_index"][None, :]).astype(int)
        E_chains = jnp.take_along_axis(E_chains, neighbours, 1)
        E_positional = self.embeddings(offset, E_chains)

        ##########################
        # define edges
        ##########################
        E = jnp.concatenate((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, neighbours


class EmbedToken(hk.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.w_init = hk.initializers.TruncatedNormal()

    @property
    def embeddings(self):
        return hk.get_parameter(
            "W_s", [self.vocab_size, self.embed_dim], init=self.w_init
        )

    def __call__(self, arr):
        if jnp.issubdtype(arr.dtype, jnp.integer):
            one_hot = jax.nn.one_hot(arr, self.vocab_size)
        else:
            one_hot = arr
        return jnp.tensordot(one_hot, self.embeddings, 1)


class ProteinMPNN(hk.Module):
    def __init__(
        self,
        num_letters,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=64,
        augment_eps=0.05,
        dropout=0.1,
        tie_messages=False
    ):
        super(ProteinMPNN, self).__init__(name="protein_mpnn")

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.tie_messages = tie_messages

        # Featurization layers
        self.features = ProteinFeatures(
            edge_features, node_features, top_k=k_neighbors, augment_eps=augment_eps
        )

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name="W_e")
        self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim)

        # Encoder layers
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name="enc" + str(i))
            for i in range(num_encoder_layers)
        ]

        # Decoder layers
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, name="dec" + str(i))
            for i in range(num_decoder_layers)
        ]
        self.W_out = hk.Linear(num_letters, with_bias=True, name="W_out")

    def __call__(self, data):
        L = data["atom_positions"].shape[0]

        # prepare node and edge embeddings
        E, neighbours = self.features(data)
        local = jnp.zeros((E.shape[0], E.shape[-1]))
        pair = self.W_e(E)

        # tie messages
        tie_info = None
        if self.tie_messages and "tie_index" in data:
            tie_info = dict(tie_index=data["tie_index"], tie_weights=data["tie_weights"])

        # encoder layers
        mask_attend = jnp.take_along_axis(
            data["mask"][:, None] * data["mask"][None, :], neighbours, 1
        )
        for layer in self.encoder_layers:
            local, pair = layer(local, pair, neighbours,
                                data["mask"], mask_attend,
                                tie_info=tie_info)
        h_V_prev = local

        def step(x, order, filled):
            autoregressive_mask = get_sep_mask(order, filled)
            mask_attend = jnp.take_along_axis(autoregressive_mask, neighbours, 1)
            mask_1D = data["mask"][:, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1 - mask_attend)

            local = h_V_prev
            h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(local), pair, neighbours)
            h_EXV_encoder = cat_neighbors_nodes(local, h_EX_encoder, neighbours)
            h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder
            h_ES = cat_neighbors_nodes(x["seq"], pair, neighbours)

            # decoder layers
            for l, layer in enumerate(self.decoder_layers):
                local = x["local"][l]
                h_ESV_decoder = cat_neighbors_nodes(local, h_ES, neighbours)
                h_ESV = mask_bw[..., None] * h_ESV_decoder + h_EXV_encoder
                local = layer(local, h_ESV, mask_V=data["mask"], tie_info=tie_info)
                # update
                x["local"] = x["local"].at[l + 1].set(local)

            logits = jax.nn.log_softmax(self.W_out(local), axis=-1)
            # set logits at unfilled positions
            x["logits"] = logits#jnp.where(filled[..., None], x["logits"], logits)

            return x

        # initial values
        sequence_one_hot = jax.nn.one_hot(data["aa"], 21, axis=-1)
        filled = data["aa"] != 20
        X = {
            "seq": jnp.where(filled[:, None], self.W_s(sequence_one_hot), 0),
            "local": jnp.array(
                [local] + [jnp.zeros_like(local)] * len(self.decoder_layers)
            ),
            "aa": sequence_one_hot,
            "logits": jnp.zeros((L, 21)),
        }

        # scan over decoding order
        # sample random decoding order in each step for filled positions
        # unfilled positions cannot attend to each other
        order = jax.random.permutation(hk.next_rng_key(), data["aa"].shape[0])
        t = order
        if t.ndim == 1:
            t = t[:, None]
        X = step(X, order, filled)
        ent = jnp.sum(-jnp.exp(X["logits"]) * X["logits"], axis=-1)
        var_ent = jnp.sum(
            jnp.exp(X["logits"]) * (X["logits"] + ent[..., None]) ** 2, axis=-1
        )

        return {"aa": X["aa"], "logits": X["logits"], "ent": ent, "var_ent": var_ent}

    def encode(self, data):
        # prepare node and edge embeddings
        E, neighbours = self.features(data)
        local = jnp.zeros((E.shape[0], E.shape[-1]))
        pair = self.W_e(E)

        # encoder layers
        mask_attend = jnp.take_along_axis(
            data["mask"][:, None] * data["mask"][None, :], neighbours, 1
        )
        for layer in self.encoder_layers:
            local, pair = layer(local, pair, neighbours, data["mask"], mask_attend)
        data["local"] = local
        data["pair"] = pair
        data["neighbours"] = neighbours
        data["logits"] = jnp.zeros((local.shape[0], 21), dtype=jnp.float32)
        return data

    def decode(self, data):
        neighbours = data["neighbours"]
        filled = data["aa"] != 20
        order = data["order"]
        ar_mask = get_sep_mask(order, filled)
        mask_attend = jnp.take_along_axis(ar_mask, neighbours, 1)
        mask_1D = data["mask"][:, None]
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1 - mask_attend)

        L = filled.shape[0]
        sequence_one_hot = jax.nn.one_hot(data["aa"], 20, axis=-1)
        x = {
            "seq": jnp.where(filled[:, None], self.W_s(sequence_one_hot), 0),
            "local": jnp.array(
                [h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)
            ),
            "aa": sequence_one_hot,
            "logits": data["logits"],
        }

        h_V = data["local"]
        h_E = data["pair"]
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, neighbours)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, neighbours)
        h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder
        h_ES = cat_neighbors_nodes(x["seq"], h_E, neighbours)

        # decoder layers
        for l, layer in enumerate(self.decoder_layers):
            h_V = x["local"][l]
            h_ESV_decoder = cat_neighbors_nodes(h_V, h_ES, neighbours)
            h_ESV = mask_bw[..., None] * h_ESV_decoder + h_EXV_encoder
            h_V = layer(h_V, h_ESV, mask_V=data["mask"])
            # update
            x["local"] = x["local"].at[l + 1].set(h_V)

        logits = jax.nn.log_softmax(self.W_out(h_V), axis=-1)
        logits = jnp.where(filled[..., None], x["logits"], logits)

        return logits


def load_params(params_path):
    return joblib.load(params_path)


def make_pmpnn(params_path, num_neighbours=48, eps=0.0):
    config = {
        "num_letters": 21,
        "node_features": 128,
        "edge_features": 128,
        "hidden_dim": 128,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "augment_eps": eps,
        "k_neighbors": num_neighbours,
        "dropout": 0,
    }

    def inner(x):
        return ProteinMPNN(**config)(x)

    run_pmpnn = hk.transform(inner).apply
    params = load_params(params_path)["model_state_dict"]
    return lambda k, d: run_pmpnn(params, k, d)


def sample_minent(mpnn, key, data):
    if "aa" not in data:
        data["aa"] = jnp.full_like(data["residue_index"], 20)
    if "order" not in data:
        data["order"] = jnp.zeros_like(data["aa"])
    data["order"] = jnp.zeros_like(data["order"])
    while (data["aa"] == 20).any():
        # run MPNN on data
        res = mpnn(key, data)
        logits = res["logits"]
        filled = data["aa"] != 20
        entropy = jnp.where(filled, jnp.inf, res["ent"])
        next_pos = jnp.argmin(entropy)
        sampled = jnp.argmax(logits[next_pos])
        data["aa"] = data["aa"].at[next_pos].set(sampled)
        data["order"] = data["order"].at[next_pos].set(np.max(data["order"] + 1))
    return data
