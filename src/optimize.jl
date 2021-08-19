export update!

function update!(n::Node, grads::NamedTuple; η = 0.01, uparams = [])
    if !isnothing(grads.params)
        n.params .+= η * grads.params
    end

    for (i, child) in enumerate(n.children)
        if child isa AbstractLeaf
            n.children[i] = update!(child, grads.children[i]; η = η, uparams = uparams)
        else
            if grads.children[i] === nothing # why is this happening?
                nothing
            else
                update!(child, grads.children[i]; η = η, uparams = uparams)
            end
        end
    end
end

function update!(n::T, grads::NamedTuple; η = 0.01, uparams = []) where {T<:AbstractLeaf}

    v = collect(params(n))
    for (i,k) in enumerate(keys(params(n)))
        if (k ∈ uparams) || isempty(uparams)
            v[i] += grads.params[k] === nothing ? zero(eltype(v)) : η * grads.params[k]
        end
    end

    return T(n.scope, (;zip(keys(params(n)), v)...))
end

function update!(n::VISumNode, grads::NamedTuple; η = 0.01, uparams = [])
    if !isnothing(grads.params)
        α = 1
        grads_KL = _Dir_KL_grad(n.params,α)
        n.params .+= η * (grads.params .- grads_KL)
    end

    for (i, child) in enumerate(n.children)
        if child isa AbstractLeaf
            n.children[i] = update!(child, grads.children[i]; η = η, uparams = uparams)
        else
            if grads.children[i] === nothing # why is this happening?
                nothing
            else
                update!(child, grads.children[i]; η = η, uparams = uparams)
            end
        end
    end
end

function _Dir_KL_obj(logβ,α) #KL objetive between Dir dist: bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    β = exp.(logβ)
    log(gamma(sum(β))) - sum(log.(gamma.(β))) + sum((β.-α).*(digamma.(β).-digamma(sum(β))))
end
