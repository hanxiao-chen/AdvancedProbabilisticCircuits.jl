export update!

function update!(n::Node, grads::NamedTuple; η = 0.01, uparams = [])
    if !isnothing(grads.params)
        n.params .+= η * grads.params
    end

    if grads.children === nothing
        return
    end

    for (i, child) in enumerate(n.children)
        if (child isa AbstractLeaf)
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

function update!(::Indicator, ::NamedTuple; η = 0.01, uparams = [])
    nothing
end

function update!(n::VISumNode, grads::NamedTuple; η = 0.01, uparams = [])
    print("VIupdate begin")
    if !isnothing(grads.params)
        α = 1
        grads_KL = _Dir_KL_grads(n.params,α)
        n.params .+= η * (grads.params .- grads_KL)
    end
    
    if grads.children === nothing
        return
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

function _Dir_KL_grads(logβ,α) #KL objetive between Dir dist: bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    grads = Array{Float64, 1}(UndefInitializer(), length(logβ))
    β0 = exp(logsumexp(logβ))
    β = exp.(logβ)
    for j in 1:length(logβ)
        tmp1 = sum([(α[i]-β[i])*trigamma(β0)*β[j] for i in 1:length(logβ) if i != j])
        tmp2 = β[j]*(digamma(β[j])-digamma(β0)) + (β[j]-α[j])*(trigamma(β[j])-trigamma(β0))*β[j]
        grads[j] = (digamma(β0)-digamma(β[j])) * β[j] + tmp1 + tmp2
    end
    return grads
end
