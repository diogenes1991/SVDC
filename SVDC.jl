module SVDC

using DSP            # Import the DSP package for signal processing
using LinearAlgebra  # Import the LinearAlgebra package for linear algebra operations

# Define a function to calculate the spectrogram of the input data using the discrete cosine transform (DCT)
function specgram(data,NFFT=256,Fs=10000)
    x = rand(ComplexF64,NFFT);   # Randomize a complex vector
    L = length(data)-NFFT        # Calculate the length of the data minus the NFFT
    ffts  = zeros(Float64,L,NFFT)   # Preallocate an LxNFFT matrix of zeros to store the power spectra
    for i in 1:L
        f = DSP.dct(data[i:i+NFFT-1])   # Compute the DCT of a segment of the input data
        ffts[i,:] = f   # Store the power spectrum in the matrix
    end
    return ffts   # Return the matrix of power spectra
end

# Define a function to compute the batch Fast Fourier Transform (FFT) of the input data
function batchFFT(data,NFFT=149,BASIS="FOURIER")
    x = rand(ComplexF64,NFFT);   # Randomize a complex vector
    L = Int64(length(data)/NFFT)   # Calculate the number of batches
    ffts  = zeros(Float64,L,NFFT)   # Preallocate an LxNFFT matrix of zeros to store the FFT coefficients
    BEG = 1   # Initialize the beginning index of each batch
    END = NFFT  # Initialize the ending index of each batch
    for i in 1:L
        if BASIS == "FOURIER"
            f = DSP.dct(data[BEG:END])   # Compute the DCT of a segment of the input data
            ffts[i,:] = f   # Store the power spectrum in the matrix
        else
            f = data[BEG:END]
            ffts[i,:] = f
        end
        BEG += NFFT   # Update the beginning and ending indices for the next batch
        END += NFFT
    end
    return ffts   # Return the matrix of FFT coefficients
end

# Define a function to compress the input audio data using singular value decomposition (SVD) of its spectrogram
function compress(data,window=1024,Fs=10000,depth=1)
    u, s, vh = svd(specgram(wav_data,window,Fs))   # Compute the SVD of the spectrogram of the input data
    return u[:,1:depth],s[1:depth],vh[1:depth,:]   # Return the compressed matrix u, vector s, and matrix v
end

# Define a function for compressing audio data using the batch FFT
function BatchCompress(data,window=149,depth=13,mode="REGULAR")
    # Compute the batch FFT of the input data
    rearr = batchFFT(data,window,mode)
    # Apply SVD to the batch FFT matrix and truncate to the desired depth
    u, s, v = svd(rearr)
    return (u[:,1:depth]*diagm(s[1:depth])),v[:,1:depth]
end

# Define a function for decompressing audio data using the batch FFT
function BatchDecompress(u,v,mode="REGULAR")
    # Compute the batch FFT matrix from the compressed data
    batch = u*v'
    decompress = zeros(size(batch)[1]*size(batch)[2],1)
    # Loop over each batch and apply the inverse FFT (or inverse DCT)
    if mode=="FOURIER"
        for i in 1:size(batch)[1]
            batch[i,:] = DSP.idct(vec(batch[i,:]))
        end
        decompress = vec(batch')
    else
        decompress = vec(batch')
    end
    # Return the decompressed audio data as a 1D vector
    return decompress
end

# Define a function for computing the fidelity between two audio data vectors
function Fidelity(data_new,data_orig)
    # Compute the root mean square error between the compressed and original data
    rval = data_new-data_orig
    rval = (sqrt(rval'*rval)/sqrt(data_orig'*data_orig))
    # Normalize by the energy of the original data to obtain a fidelity measure between 0 and 1
    rval = 1.0 - rval[1,1]
    return rval
end

function f_fidelity(original,compressed)
    d_o = DSP.dct(original)
    d_c = DSP.dct(compressed)
    return Fidelity(d_o,d_c)
end

#ITU-R 468 noise weighting
function R_ITU(f::Float64)
    h1 = -4.737338981378384e-24 * f^6 + 2.043828333606125e-15 * f^4 - 1.363894795463638e-7 * f^2 + 1
    h2 = 1.306612257412824e-19 * f^5 - 2.118150887518656e-11 * f^3 + 5.559488023498642e-4 * f
    num = 1.246332637532143e-4 * f
    den = sqrt(h1^2 + h2^2)
    return num / den
end
#source: https://en.m.wikipedia.org/wiki/ITU-R_468_noise_weighting#Summary_of_specification

function wf_fidelity(original,compressed,fs) #weighted
    d_o    = DSP.dct(original)
    d_c    = DSP.dct(compressed)
    l      = length(d_o)
    freq   = [0:fs/l:fs]
    metric = sqrt(R_ITU(freq))
    dif    = (d_o - d_c).*metric
    dom    = d_o.*metric
    return 1-(dif'*dif)/(dom'*dom)
end

function Compute(wav,window,depth,mode,weighted,fs)
    u, v = BatchCompress(wav,window,depth,mode)
        
    # Reconstruct the audio data with compressed data and Fourier basis
    comp = BatchDecompress(u,v,mode)
        
    # Compute sizes of u and v' and store them in su and sv
    su = size(u)
    sv = size(v')
        
    # Compute compression factor and fidelity
    C = (1.0 - (su[1]*su[2] + sv[1]*sv[2]) / (su[1]*sv[2]))
    F = 0
    if weighted == true
        F = wf_fidelity(wav,comp,fs)
    else 
        F = Fidelity(comp, wav)
    end
    return F,C
end

function Process(wav,window,slices,mode,weighted,fs)
    
    # Initialize matrices to store results
    fids = zeros(slices,2)
    cfac = zeros(slices,2)
    metr = zeros(slices,2)
    
    # Loop over different compression depths
    for i in 1:slices
        
        # Compress the audio data with depth i and Fourier basis
        F,C = Compute(wav,window,i,mode,weighted,fs)
        
        # Compute compression factor and store it in cfac matrix
        cfac[i, 1] = i/slices
        cfac[i, 2] = C
    
        # Compute fidelity between compressed and original data and store it in fids matrix
        fids[i, 1] = i/slices
        fids[i, 2] = F
        
        # Compute metric as the product of fidelity and compression factor and store it in metr matrix
        metr[i, 1] = C
        metr[i, 2] = F
    end
    
    return cfac,fids,metr
end

end