#include <metal_stdlib>
using namespace metal;

struct QuantizationConstants {
    uint count;
    float quantizationStep;
};

struct PadPlaneConstants {
    uint sourceWidth;
    uint sourceHeight;
    uint paddedWidth;
    uint paddedHeight;
    uint channel;
};

struct CropPlaneConstants {
    uint paddedWidth;
    uint outputWidth;
    uint outputHeight;
};

struct CropNV12Constants {
    uint yPaddedWidth;
    uint chromaPaddedWidth;
    uint outputWidth;
    uint outputHeight;
};

struct PlaneQuantizationDescriptor {
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
    uint quantCode;
    float baseScale;
};

struct PlaneQuantizationConstants {
    uint descriptorCount;
};

struct SparseCoefficientEntry {
    uint destinationOffset;
    int coefficient;
    uint quantCode;
    uint qScaleCode;
};

struct SparseApplyConstants {
    uint entryCount;
    uint sampleCount;
};

struct SparsePacketDecodeDescriptor {
    uint packetOffset;
    uint payloadEnd;
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
};

struct SparsePacketDecodeConstants {
    uint descriptorCount;
    uint sampleCount;
};

struct RateControlStatsDescriptor {
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
    uint quantCode;
    uint qScaleCode;
    float distortionScale;
};

struct RateControlQuantStats {
    float squareError;
    uint encodeCostBits;
};

struct RateControlStatsConstants {
    uint descriptorCount;
};

struct PacketByteCostDescriptor {
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
};

struct PacketByteCostConstants {
    uint descriptorCount;
};

struct SparsePacketEncodeDescriptor {
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
    uint blockIndex;
    uint quantLevel;
    uint sequence;
    uint quantCode;
};

struct SparsePacketEncodeConstants {
    uint descriptorCount;
    uint maxPacketBytes;
};

struct RateControlBucketConstants {
    uint blockCount;
};

struct RateControlBucketSavingsConstants {
    uint blockCount;
};

struct DWTConstants {
    uint activeWidth;
    uint activeHeight;
    uint stride;
    uint phase;
};

constant float dwtAlpha = -1.586134342059924f;
constant float dwtBeta = -0.052980118572961f;
constant float dwtGamma = 0.882911075530934f;
constant float dwtDelta = 0.443506852043971f;
constant float dwtK = 1.230174104914001f;
constant float dwtInvK = 1.0f / 1.230174104914001f;

static inline int mirrorIndex(int index, int count) {
    if (count <= 1) {
        return 0;
    }
    if (index < 0) {
        return -index;
    }
    if (index >= count) {
        return 2 * count - index - 2;
    }
    return index;
}

kernel void pyrowave_pad_plane(
    device const uchar *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant PadPlaneConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.paddedWidth || y >= constants.paddedHeight) {
        return;
    }

    uint sourceX = min(x, constants.sourceWidth - 1u);
    uint sourceY = min(y, constants.sourceHeight - 1u);
    output[y * constants.paddedWidth + x] = float(input[sourceY * constants.sourceWidth + sourceX]) / 255.0f - 0.5f;
}

kernel void pyrowave_pad_texture_plane(
    texture2d<float, access::read> input [[texture(0)]],
    device float *output [[buffer(0)]],
    constant PadPlaneConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.paddedWidth || y >= constants.paddedHeight) {
        return;
    }

    uint sourceX = min(x, constants.sourceWidth - 1u);
    uint sourceY = min(y, constants.sourceHeight - 1u);
    float4 value = input.read(uint2(sourceX, sourceY));
    float sample = constants.channel == 1u ? value.g : value.r;
    output[y * constants.paddedWidth + x] = sample - 0.5f;
}

kernel void pyrowave_crop_plane(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant CropPlaneConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.outputWidth || y >= constants.outputHeight) {
        return;
    }

    float normalized = clamp(input[y * constants.paddedWidth + x] + 0.5f, 0.0f, 1.0f);
    output[y * constants.outputWidth + x] = uchar(round(normalized * 255.0f));
}

kernel void pyrowave_crop_texture_plane(
    device const float *input [[buffer(0)]],
    texture2d<float, access::write> output [[texture(0)]],
    constant CropPlaneConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.outputWidth || y >= constants.outputHeight) {
        return;
    }

    float normalized = clamp(input[y * constants.paddedWidth + x] + 0.5f, 0.0f, 1.0f);
    output.write(float4(normalized, 0.0f, 0.0f, 1.0f), uint2(x, y));
}

kernel void pyrowave_crop_nv12_textures(
    device const float *yInput [[buffer(0)]],
    device const float *cbInput [[buffer(1)]],
    device const float *crInput [[buffer(2)]],
    texture2d<float, access::write> yOutput [[texture(0)]],
    texture2d<float, access::write> cbCrOutput [[texture(1)]],
    constant CropNV12Constants &constants [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x < constants.outputWidth && y < constants.outputHeight) {
        float luma = clamp(yInput[y * constants.yPaddedWidth + x] + 0.5f, 0.0f, 1.0f);
        yOutput.write(float4(luma, 0.0f, 0.0f, 1.0f), uint2(x, y));
    }

    uint chromaWidth = constants.outputWidth / 2u;
    uint chromaHeight = constants.outputHeight / 2u;
    if (x < chromaWidth && y < chromaHeight) {
        uint index = y * constants.chromaPaddedWidth + x;
        float cb = clamp(cbInput[index] + 0.5f, 0.0f, 1.0f);
        float cr = clamp(crInput[index] + 0.5f, 0.0f, 1.0f);
        cbCrOutput.write(float4(cb, cr, 0.0f, 1.0f), uint2(x, y));
    }
}

kernel void pyrowave_quantize(
    device const float *input [[buffer(0)]],
    device short *output [[buffer(1)]],
    constant QuantizationConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.count) {
        return;
    }

    float scaled = rint(input[index] / constants.quantizationStep);
    scaled = clamp(scaled, -32768.0f, 32767.0f);
    output[index] = short(scaled);
}

kernel void pyrowave_dequantize(
    device const short *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant QuantizationConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.count) {
        return;
    }

    output[index] = float(input[index]) * constants.quantizationStep;
}

static inline uchar pyrowave_encode_8x8_scale(float scale) {
    float encoded = ceil((scale - 0.25f) * 8.0f);
    encoded = clamp(encoded, 0.0f, 15.0f);
    return uchar(encoded);
}

static inline uchar pyrowave_encode_8x8_scale_code(float maxScaledCoefficient) {
    if (maxScaledCoefficient < 1.0f) {
        return 6;
    }

    int exponent = int(floor(log2(maxScaledCoefficient - 0.25f))) + 1;
    float targetMax = exp2(float(exponent)) - 0.25f;
    return pyrowave_encode_8x8_scale(maxScaledCoefficient / targetMax);
}

static inline float pyrowave_decode_block_scale(uint quantCode) {
    int exponent = 4 - int(quantCode >> 3u);
    uint mantissa = quantCode & 7u;
    uint scaleShift = uint(20 + exponent);
    return float((8u + mantissa) * (1u << scaleShift)) / (8.0f * 1024.0f * 1024.0f);
}

static inline float pyrowave_decode_8x8_scale(uint code) {
    return float(code) / 8.0f + 0.25f;
}

static inline uint pyrowave_significant_bit_count(uint value) {
    uint bits = 0u;
    while (value != 0u) {
        value >>= 1u;
        bits += 1u;
    }
    return bits;
}

static inline uint2 pyrowave_coordinate_in_8x8(uint subblock, uint pixel) {
    return uint2((subblock / 4u) * 4u + (pixel >> 1u), (subblock % 4u) * 2u + (pixel & 1u));
}

kernel void pyrowave_quantize_plane_tiles(
    device const float *samples [[buffer(0)]],
    device short *coefficients [[buffer(1)]],
    device const PlaneQuantizationDescriptor *descriptors [[buffer(2)]],
    device uchar *qScaleCodes [[buffer(3)]],
    constant PlaneQuantizationConstants &constants [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint descriptorIndex = index >> 4u;
    uint smallBlock = index & 15u;
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    PlaneQuantizationDescriptor descriptor = descriptors[descriptorIndex];
    uint smallBlockX = smallBlock & 3u;
    uint smallBlockY = smallBlock >> 2u;
    uint localX0 = smallBlockX * 8u;
    uint localY0 = smallBlockY * 8u;
    uint validWidth = descriptor.validWidth > localX0 ? min(8u, descriptor.validWidth - localX0) : 0u;
    uint validHeight = descriptor.validHeight > localY0 ? min(8u, descriptor.validHeight - localY0) : 0u;
    if (validWidth == 0u || validHeight == 0u) {
        qScaleCodes[descriptorIndex * 16u + smallBlock] = 6;
        return;
    }

    uint originX = descriptor.originX + localX0;
    uint originY = descriptor.originY + localY0;
    float maxScaledCoefficient = 0.0f;
    for (uint y = 0; y < validHeight; ++y) {
        uint row = (originY + y) * descriptor.stride + originX;
        for (uint x = 0; x < validWidth; ++x) {
            maxScaledCoefficient = max(maxScaledCoefficient, fabs(samples[row + x] * descriptor.baseScale));
        }
    }

    uchar qScaleCode = pyrowave_encode_8x8_scale_code(maxScaledCoefficient);
    qScaleCodes[descriptorIndex * 16u + smallBlock] = qScaleCode;
    float decoded8x8Scale = float(qScaleCode) / 8.0f + 0.25f;
    float quantScale = 1.0f / decoded8x8Scale;

    for (uint y = 0; y < validHeight; ++y) {
        uint row = (originY + y) * descriptor.stride + originX;
        for (uint x = 0; x < validWidth; ++x) {
            float scaled = trunc(samples[row + x] * descriptor.baseScale * quantScale);
            scaled = clamp(scaled, -32768.0f, 32767.0f);
            coefficients[row + x] = short(scaled);
        }
    }
}

kernel void pyrowave_apply_sparse_coefficients(
    device float *samples [[buffer(0)]],
    device const SparseCoefficientEntry *entries [[buffer(1)]],
    constant SparseApplyConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.entryCount) {
        return;
    }

    SparseCoefficientEntry entry = entries[index];
    if (entry.destinationOffset >= constants.sampleCount) {
        return;
    }

    float value = float(entry.coefficient);
    if (value > 0.0f) {
        value += 0.5f;
    } else if (value < 0.0f) {
        value -= 0.5f;
    }
    samples[entry.destinationOffset] = value
        * pyrowave_decode_block_scale(entry.quantCode)
        * pyrowave_decode_8x8_scale(entry.qScaleCode);
}

kernel void pyrowave_rate_control_tile_stats(
    device const short *coefficients [[buffer(0)]],
    device const RateControlStatsDescriptor *descriptors [[buffer(1)]],
    device uint *numPlanes [[buffer(2)]],
    device RateControlQuantStats *stats [[buffer(3)]],
    constant RateControlStatsConstants &constants [[buffer(4)]],
    uint descriptorIndex [[thread_position_in_grid]]
) {
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    RateControlStatsDescriptor descriptor = descriptors[descriptorIndex];
    uint statsOffset = descriptorIndex * 15u;
    if (descriptor.validWidth == 0u || descriptor.validHeight == 0u) {
        numPlanes[descriptorIndex] = 0u;
        for (uint quantLevel = 0u; quantLevel < 15u; ++quantLevel) {
            stats[statsOffset + quantLevel] = RateControlQuantStats{0.0f, 0u};
        }
        return;
    }

    uint maximumMagnitude = 0u;
    for (uint y = 0u; y < descriptor.validHeight; ++y) {
        uint row = (descriptor.originY + y) * descriptor.stride + descriptor.originX;
        for (uint x = 0u; x < descriptor.validWidth; ++x) {
            int coefficient = int(coefficients[row + x]);
            uint magnitude = uint(abs(coefficient));
            maximumMagnitude = max(maximumMagnitude, magnitude);
        }
    }

    numPlanes[descriptorIndex] = maximumMagnitude == 0u ? 0u : min(14u, pyrowave_significant_bit_count(maximumMagnitude));
    float coefficientToSampleScale = pyrowave_decode_block_scale(descriptor.quantCode)
        * pyrowave_decode_8x8_scale(descriptor.qScaleCode);
    float distortionWeight = coefficientToSampleScale * coefficientToSampleScale * descriptor.distortionScale;

    for (uint quantLevel = 0u; quantLevel < 15u; ++quantLevel) {
        float squareError = 0.0f;
        uint encodeCostBits = 0u;
        uint retainedValues = 0u;

        for (uint y = 0u; y < descriptor.validHeight; ++y) {
            uint row = (descriptor.originY + y) * descriptor.stride + descriptor.originX;
            for (uint x = 0u; x < descriptor.validWidth; ++x) {
                int coefficient = int(coefficients[row + x]);
                uint magnitude = uint(abs(coefficient));
                uint retainedMagnitude = magnitude >> quantLevel;
                if (retainedMagnitude != 0u) {
                    retainedValues += 1u;
                    encodeCostBits += pyrowave_significant_bit_count(retainedMagnitude);
                    if (quantLevel != 0u) {
                        float reconstructedMagnitude = (float(retainedMagnitude) + 0.5f) * float(1u << quantLevel);
                        float delta = float(magnitude) - reconstructedMagnitude;
                        squareError += delta * delta * distortionWeight;
                    }
                } else {
                    squareError += float(magnitude * magnitude) * distortionWeight;
                }
            }
        }

        if (retainedValues != 0u) {
            encodeCostBits += retainedValues;
        }
        stats[statsOffset + quantLevel] = RateControlQuantStats{squareError, encodeCostBits};
    }
}

kernel void pyrowave_packet_byte_costs(
    device const short *coefficients [[buffer(0)]],
    device const PacketByteCostDescriptor *descriptors [[buffer(1)]],
    device uint *byteCosts [[buffer(2)]],
    constant PacketByteCostConstants &constants [[buffer(3)]],
    uint descriptorIndex [[thread_position_in_grid]]
) {
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    PacketByteCostDescriptor descriptor = descriptors[descriptorIndex];
    uint outputOffset = descriptorIndex * 15u;
    for (uint quantLevel = 0u; quantLevel < 15u; ++quantLevel) {
        bool activeSmallBlocks[16];
        uint activeBlockCount = 0u;
        for (uint smallBlock = 0u; smallBlock < 16u; ++smallBlock) {
            activeSmallBlocks[smallBlock] = false;
        }

        for (uint y = 0u; y < descriptor.validHeight; ++y) {
            uint row = (descriptor.originY + y) * descriptor.stride + descriptor.originX;
            for (uint x = 0u; x < descriptor.validWidth; ++x) {
                int value = int(coefficients[row + x]);
                uint magnitude = uint(abs(value)) >> quantLevel;
                if (magnitude != 0u) {
                    uint smallBlock = (y / 8u) * 4u + (x / 8u);
                    if (!activeSmallBlocks[smallBlock]) {
                        activeSmallBlocks[smallBlock] = true;
                        activeBlockCount += 1u;
                    }
                }
            }
        }

        if (activeBlockCount == 0u) {
            byteCosts[outputOffset + quantLevel] = 0u;
            continue;
        }

        uint magnitudePayloadBytes = 0u;
        uint signCount = 0u;

        for (uint smallBlock = 0u; smallBlock < 16u; ++smallBlock) {
            if (!activeSmallBlocks[smallBlock]) {
                continue;
            }

            uint smallOriginX = (smallBlock % 4u) * 8u;
            uint smallOriginY = (smallBlock / 4u) * 8u;
            uint bitWidths[8];
            uint maxBitWidth = 0u;

            for (uint subblock = 0u; subblock < 8u; ++subblock) {
                uint maxMagnitude = 0u;
                for (uint pixel = 0u; pixel < 8u; ++pixel) {
                    uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
                    uint x = smallOriginX + coord.x;
                    uint y = smallOriginY + coord.y;
                    uint magnitude = 0u;
                    if (x < descriptor.validWidth && y < descriptor.validHeight) {
                        uint index = (descriptor.originY + y) * descriptor.stride + descriptor.originX + x;
                        magnitude = uint(abs(int(coefficients[index]))) >> quantLevel;
                    }
                    maxMagnitude = max(maxMagnitude, magnitude);
                    if (magnitude != 0u) {
                        signCount += 1u;
                    }
                }
                uint width = pyrowave_significant_bit_count(maxMagnitude);
                bitWidths[subblock] = width;
                maxBitWidth = max(maxBitWidth, width);
            }

            uint basePlanes = maxBitWidth > 3u ? maxBitWidth - 3u : 0u;
            for (uint subblock = 0u; subblock < 8u; ++subblock) {
                magnitudePayloadBytes += max(bitWidths[subblock], basePlanes);
            }
        }

        uint signPayloadBytes = (signCount + 7u) / 8u;
        uint unpaddedSize = 8u + activeBlockCount * 2u + activeBlockCount + magnitudePayloadBytes + signPayloadBytes;
        byteCosts[outputOffset + quantLevel] = ((unpaddedSize + 3u) / 4u) * 4u;
    }
}

kernel void pyrowave_packet_byte_costs_smallblocks(
    device const short *coefficients [[buffer(0)]],
    device const PacketByteCostDescriptor *descriptors [[buffer(1)]],
    device atomic_uint *byteCostPartials [[buffer(2)]],
    device atomic_uint *signCounts [[buffer(3)]],
    constant PacketByteCostConstants &constants [[buffer(4)]],
    uint workIndex [[thread_position_in_grid]]
) {
    uint descriptorIndex = workIndex >> 4u;
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    uint smallBlock = workIndex & 15u;
    PacketByteCostDescriptor descriptor = descriptors[descriptorIndex];
    uint smallOriginX = (smallBlock % 4u) * 8u;
    uint smallOriginY = (smallBlock / 4u) * 8u;
    if (smallOriginX >= descriptor.validWidth || smallOriginY >= descriptor.validHeight) {
        return;
    }

    uint outputOffset = descriptorIndex * 15u;
    for (uint quantLevel = 0u; quantLevel < 15u; ++quantLevel) {
        uint bitWidths[8];
        uint maxBitWidth = 0u;
        uint signCount = 0u;

        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            uint maxMagnitude = 0u;
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
                uint x = smallOriginX + coord.x;
                uint y = smallOriginY + coord.y;
                uint magnitude = 0u;
                if (x < descriptor.validWidth && y < descriptor.validHeight) {
                    uint index = (descriptor.originY + y) * descriptor.stride + descriptor.originX + x;
                    int raw = int(coefficients[index]);
                    magnitude = (raw < 0 ? uint(-raw) : uint(raw)) >> quantLevel;
                }
                maxMagnitude = max(maxMagnitude, magnitude);
                if (magnitude != 0u) {
                    signCount += 1u;
                }
            }
            uint width = pyrowave_significant_bit_count(maxMagnitude);
            bitWidths[subblock] = width;
            maxBitWidth = max(maxBitWidth, width);
        }

        if (maxBitWidth == 0u) {
            continue;
        }

        uint basePlanes = maxBitWidth > 3u ? maxBitWidth - 3u : 0u;
        uint magnitudePayloadBytes = 0u;
        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            magnitudePayloadBytes += max(bitWidths[subblock], basePlanes);
        }

        uint candidateOffset = outputOffset + quantLevel;
        atomic_fetch_add_explicit(&byteCostPartials[candidateOffset], 3u + magnitudePayloadBytes, memory_order_relaxed);
        atomic_fetch_add_explicit(&signCounts[candidateOffset], signCount, memory_order_relaxed);
    }
}

kernel void pyrowave_packet_byte_costs_finalize(
    device uint *byteCosts [[buffer(0)]],
    device const uint *signCounts [[buffer(1)]],
    constant PacketByteCostConstants &constants [[buffer(2)]],
    uint candidateIndex [[thread_position_in_grid]]
) {
    uint candidateCount = constants.descriptorCount * 15u;
    if (candidateIndex >= candidateCount) {
        return;
    }

    uint partialBytes = byteCosts[candidateIndex];
    if (partialBytes == 0u) {
        byteCosts[candidateIndex] = 0u;
        return;
    }

    uint signPayloadBytes = (signCounts[candidateIndex] + 7u) / 8u;
    uint unpaddedSize = 8u + partialBytes + signPayloadBytes;
    byteCosts[candidateIndex] = ((unpaddedSize + 3u) / 4u) * 4u;
}

static inline void pyrowave_write_u16_le(device uchar *output, uint offset, uint value) {
    output[offset + 0u] = uchar(value & 0xffu);
    output[offset + 1u] = uchar((value >> 8u) & 0xffu);
}

static inline void pyrowave_write_u32_le(device uchar *output, uint offset, uint value) {
    output[offset + 0u] = uchar(value & 0xffu);
    output[offset + 1u] = uchar((value >> 8u) & 0xffu);
    output[offset + 2u] = uchar((value >> 16u) & 0xffu);
    output[offset + 3u] = uchar((value >> 24u) & 0xffu);
}

static inline uint pyrowave_read_u16_le(device const uchar *input, uint offset) {
    return uint(input[offset]) | (uint(input[offset + 1u]) << 8u);
}

static inline uint pyrowave_read_u32_le(device const uchar *input, uint offset) {
    return uint(input[offset])
        | (uint(input[offset + 1u]) << 8u)
        | (uint(input[offset + 2u]) << 16u)
        | (uint(input[offset + 3u]) << 24u);
}

static inline uint pyrowave_modify_quant_code(uint quantCode, uint quantLevel) {
    if (quantLevel == 0u) {
        return quantCode;
    }
    uint exponent = quantCode >> 3u;
    exponent = exponent > quantLevel ? exponent - quantLevel : 0u;
    return (exponent << 3u) | (quantCode & 7u);
}

static inline short pyrowave_quantized_packet_value(
    device const short *coefficients,
    SparsePacketEncodeDescriptor descriptor,
    uint x,
    uint y
) {
    if (x >= descriptor.validWidth || y >= descriptor.validHeight) {
        return 0;
    }

    int raw = int(coefficients[(descriptor.originY + y) * descriptor.stride + descriptor.originX + x]);
    uint magnitude = raw < 0 ? uint(-raw) : uint(raw);
    magnitude >>= descriptor.quantLevel;
    if (magnitude == 0u) {
        return 0;
    }
    return raw < 0 ? short(-int(magnitude)) : short(magnitude);
}

kernel void pyrowave_encode_sparse_packets(
    device const short *coefficients [[buffer(0)]],
    device const SparsePacketEncodeDescriptor *descriptors [[buffer(1)]],
    device const uchar *qScaleCodes [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    device uint *outputSizes [[buffer(4)]],
    constant SparsePacketEncodeConstants &constants [[buffer(5)]],
    uint descriptorIndex [[thread_position_in_grid]]
) {
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    SparsePacketEncodeDescriptor descriptor = descriptors[descriptorIndex];
    device uchar *packet = output + descriptorIndex * constants.maxPacketBytes;
    outputSizes[descriptorIndex] = 0u;

    uint ballot = 0u;
    for (uint y = 0u; y < descriptor.validHeight; ++y) {
        for (uint x = 0u; x < descriptor.validWidth; ++x) {
            short value = pyrowave_quantized_packet_value(coefficients, descriptor, x, y);
            if (value != 0) {
                uint smallBlock = (y / 8u) * 4u + (x / 8u);
                ballot |= 1u << smallBlock;
            }
        }
    }
    if (ballot == 0u) {
        return;
    }

    uint activeBlockCount = popcount(ballot);
    uint codeWordStart = 8u;
    uint qScaleStart = codeWordStart + activeBlockCount * 2u;
    uint magnitudeStart = qScaleStart + activeBlockCount;
    uint magnitudeOffset = 0u;
    uchar signPayload[128];
    for (uint i = 0u; i < 128u; ++i) {
        signPayload[i] = 0;
    }
    uint signCount = 0u;
    uint compactIndex = 0u;
    device const uchar *descriptorQScaleCodes = qScaleCodes + descriptorIndex * 16u;

    for (uint smallBlock = 0u; smallBlock < 16u; ++smallBlock) {
        if ((ballot & (1u << smallBlock)) == 0u) {
            continue;
        }

        uint smallOriginX = (smallBlock % 4u) * 8u;
        uint smallOriginY = (smallBlock / 4u) * 8u;
        uint bitWidths[8];
        uint maxBitWidth = 0u;

        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            uint maxMagnitude = 0u;
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
                short value = pyrowave_quantized_packet_value(coefficients, descriptor, smallOriginX + coord.x, smallOriginY + coord.y);
                int raw = int(value);
                uint magnitude = raw < 0 ? uint(-raw) : uint(raw);
                maxMagnitude = max(maxMagnitude, magnitude);
            }
            bitWidths[subblock] = pyrowave_significant_bit_count(maxMagnitude);
            maxBitWidth = max(maxBitWidth, bitWidths[subblock]);
        }

        uint basePlanes = maxBitWidth > 3u ? maxBitWidth - 3u : 0u;
        uint codeWord = 0u;
        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            uint encodedPlanes = max(bitWidths[subblock], basePlanes);
            uint twoBitCode = min(3u, encodedPlanes > basePlanes ? encodedPlanes - basePlanes : 0u);
            codeWord |= twoBitCode << (2u * subblock);

            for (uint plane = 0u; plane < encodedPlanes; ++plane) {
                uint bit = encodedPlanes - plane - 1u;
                uchar byte = 0;
                for (uint pixel = 0u; pixel < 8u; ++pixel) {
                    uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
                    short value = pyrowave_quantized_packet_value(coefficients, descriptor, smallOriginX + coord.x, smallOriginY + coord.y);
                    int raw = int(value);
                    uint magnitude = raw < 0 ? uint(-raw) : uint(raw);
                    if (((magnitude >> bit) & 1u) != 0u) {
                        byte |= uchar(1u << pixel);
                    }
                }
                packet[magnitudeStart + magnitudeOffset] = byte;
                magnitudeOffset += 1u;
            }
        }

        pyrowave_write_u16_le(packet, codeWordStart + compactIndex * 2u, codeWord);
        packet[qScaleStart + compactIndex] = uchar((uint(descriptorQScaleCodes[smallBlock]) << 4u) | (basePlanes & 0x0fu));

        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
                short value = pyrowave_quantized_packet_value(coefficients, descriptor, smallOriginX + coord.x, smallOriginY + coord.y);
                if (value != 0) {
                    if (value < 0) {
                        signPayload[signCount / 8u] |= uchar(1u << (signCount & 7u));
                    }
                    signCount += 1u;
                }
            }
        }

        compactIndex += 1u;
    }

    uint signByteCount = (signCount + 7u) / 8u;
    uint signStart = magnitudeStart + magnitudeOffset;
    for (uint i = 0u; i < signByteCount; ++i) {
        packet[signStart + i] = signPayload[i];
    }

    uint unpaddedSize = signStart + signByteCount;
    uint payloadWords = (unpaddedSize + 3u) / 4u;
    uint packetSize = payloadWords * 4u;
    if (packetSize > constants.maxPacketBytes) {
        return;
    }

    pyrowave_write_u16_le(packet, 0u, ballot);
    uint packedPayload = (payloadWords & 0x0fffu) | ((descriptor.sequence & 7u) << 12u);
    pyrowave_write_u16_le(packet, 2u, packedPayload);
    uint packedBlock = (descriptor.blockIndex << 8u) | pyrowave_modify_quant_code(descriptor.quantCode, descriptor.quantLevel);
    pyrowave_write_u32_le(packet, 4u, packedBlock);
    outputSizes[descriptorIndex] = packetSize;
}

kernel void pyrowave_decode_sparse_packets(
    device float *samples [[buffer(0)]],
    device const uchar *packets [[buffer(1)]],
    device const SparsePacketDecodeDescriptor *descriptors [[buffer(2)]],
    constant SparsePacketDecodeConstants &constants [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint descriptorIndex = index >> 4u;
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }
    uint targetSmallBlock = index & 15u;

    SparsePacketDecodeDescriptor descriptor = descriptors[descriptorIndex];
    uint packetOffset = descriptor.packetOffset;
    uint ballot = pyrowave_read_u16_le(packets, packetOffset);
    if (ballot == 0u || descriptor.validWidth == 0u || descriptor.validHeight == 0u) {
        return;
    }
    if ((ballot & (1u << targetSmallBlock)) == 0u) {
        return;
    }

    uint packedBlock = pyrowave_read_u32_le(packets, packetOffset + 4u);
    uint quantCode = packedBlock & 0xffu;
    uint activeBlockCount = popcount(ballot);
    uint codeWordStart = packetOffset + 8u;
    uint qScaleStart = codeWordStart + activeBlockCount * 2u;
    uint magnitudeStart = qScaleStart + activeBlockCount;

    uint magnitudeOffset = 0u;
    uint signBase = 0u;
    uint totalMagnitudeBytes = 0u;
    uint targetCompactIndex = 0u;
    uint targetMagnitudeOffset = 0u;
    uint targetSignBase = 0u;
    uint compactIndex = 0u;
    for (uint smallBlock = 0u; smallBlock < 16u; ++smallBlock) {
        if ((ballot & (1u << smallBlock)) == 0u) {
            continue;
        }

        uint codeWord = pyrowave_read_u16_le(packets, codeWordStart + compactIndex * 2u);
        uint qScale = uint(packets[qScaleStart + compactIndex]);
        uint basePlanes = qScale & 0x0fu;
        uint smallBlockMagnitudeOffset = magnitudeOffset;
        uint smallBlockMagnitudeBytes = 0u;
        uint smallBlockSignCount = 0u;
        for (uint subblock = 0u; subblock < 8u; ++subblock) {
            uint encodedPlanes = ((codeWord >> (2u * subblock)) & 0x3u) + basePlanes;
            smallBlockMagnitudeBytes += encodedPlanes;
            uint magnitudes[8];
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                magnitudes[pixel] = 0u;
            }
            uint subblockOffset = smallBlockMagnitudeOffset;
            for (uint previousSubblock = 0u; previousSubblock < subblock; ++previousSubblock) {
                subblockOffset += ((codeWord >> (2u * previousSubblock)) & 0x3u) + basePlanes;
            }
            for (uint plane = 0u; plane < encodedPlanes && magnitudeStart + subblockOffset + plane < descriptor.payloadEnd; ++plane) {
                uint byte = uint(packets[magnitudeStart + subblockOffset + plane]);
                for (uint pixel = 0u; pixel < 8u; ++pixel) {
                    magnitudes[pixel] = (magnitudes[pixel] << 1u) | ((byte >> pixel) & 1u);
                }
            }
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                if (magnitudes[pixel] != 0u) {
                    smallBlockSignCount += 1u;
                }
            }
        }

        if (smallBlock == targetSmallBlock) {
            targetCompactIndex = compactIndex;
            targetMagnitudeOffset = magnitudeOffset;
            targetSignBase = signBase;
        }
        magnitudeOffset += smallBlockMagnitudeBytes;
        signBase += smallBlockSignCount;
        totalMagnitudeBytes += smallBlockMagnitudeBytes;
        compactIndex += 1u;
    }

    uint signStart = magnitudeStart + totalMagnitudeBytes;
    uint cursor = magnitudeStart + targetMagnitudeOffset;
    uint signIndex = targetSignBase;
    uint smallOriginX = (targetSmallBlock % 4u) * 8u;
    uint smallOriginY = (targetSmallBlock / 4u) * 8u;
    uint codeWord = pyrowave_read_u16_le(packets, codeWordStart + targetCompactIndex * 2u);
    uint qScale = uint(packets[qScaleStart + targetCompactIndex]);
    uint basePlanes = qScale & 0x0fu;
    uint qScaleCode = qScale >> 4u;
    float coefficientToSampleScale = pyrowave_decode_block_scale(quantCode)
        * pyrowave_decode_8x8_scale(qScaleCode);

    for (uint subblock = 0u; subblock < 8u; ++subblock) {
        uint encodedPlanes = ((codeWord >> (2u * subblock)) & 0x3u) + basePlanes;
        uint magnitudes[8];
        for (uint pixel = 0u; pixel < 8u; ++pixel) {
            magnitudes[pixel] = 0u;
        }
        for (uint plane = 0u; plane < encodedPlanes && cursor < descriptor.payloadEnd; ++plane) {
            uint byte = uint(packets[cursor]);
            cursor += 1u;
            for (uint pixel = 0u; pixel < 8u; ++pixel) {
                magnitudes[pixel] = (magnitudes[pixel] << 1u) | ((byte >> pixel) & 1u);
            }
        }

        for (uint pixel = 0u; pixel < 8u; ++pixel) {
            uint magnitude = magnitudes[pixel];
            if (magnitude == 0u) {
                continue;
            }

            uint2 coord = pyrowave_coordinate_in_8x8(subblock, pixel);
            uint localX = smallOriginX + coord.x;
            uint localY = smallOriginY + coord.y;
            if (localX < descriptor.validWidth && localY < descriptor.validHeight) {
                uint destinationOffset = (descriptor.originY + localY) * descriptor.stride + descriptor.originX + localX;
                if (destinationOffset < constants.sampleCount) {
                    uint signByte = uint(packets[signStart + signIndex / 8u]);
                    bool negative = ((signByte >> (signIndex & 7u)) & 1u) != 0u;
                    float value = float(magnitude);
                    value = negative ? -(value + 0.5f) : value + 0.5f;
                    samples[destinationOffset] = value * coefficientToSampleScale;
                }
            }
            signIndex += 1u;
        }
    }
}

kernel void pyrowave_rate_control_bucket_indices(
    device const float *distortions [[buffer(0)]],
    device const uint *packetByteCosts [[buffer(1)]],
    device uint *bucketIndices [[buffer(2)]],
    constant RateControlBucketConstants &constants [[buffer(3)]],
    uint blockIndex [[thread_position_in_grid]]
) {
    if (blockIndex >= constants.blockCount) {
        return;
    }

    constexpr uint candidateCount = 15u;
    constexpr uint bucketCount = 128u;
    constexpr uint bucketClusterWidth = 16u;
    uint offset = blockIndex * candidateCount;
    float baseDistortion = distortions[offset];
    uint baseCost = packetByteCosts[offset];
    uint raw[candidateCount];

    for (uint quantLevel = 0u; quantLevel < candidateCount; ++quantLevel) {
        uint bucket = 0u;
        if (quantLevel != 0u && packetByteCosts[offset + quantLevel] != baseCost) {
            float distortionDelta = max(distortions[offset + quantLevel] - baseDistortion, 0.0f);
            int saving = int(baseCost) - int(packetByteCosts[offset + quantLevel]);
            float costSaving = saving > 0 ? float(saving) : 1.40129846e-45f;
            float index = 60.0f + 2.0f * log2(distortionDelta / costSaving);
            if (isfinite(index)) {
                bucket = uint(clamp(floor(index + 0.5f), 0.0f, 127.0f));
            }
        }
        raw[quantLevel] = min(bucket, bucketCount - bucketClusterWidth + quantLevel);
    }

    for (uint quantLevel = 0u; quantLevel < candidateCount; ++quantLevel) {
        uint bucket = raw[quantLevel];
        for (uint previous = 0u; previous < quantLevel; ++previous) {
            bucket = max(bucket, raw[previous] + quantLevel - previous);
        }
        bucketIndices[offset + quantLevel] = min(bucketCount - 1u, bucket);
    }
}

static inline float pyrowave_quantized_square_error(float squareError) {
    return float(half(clamp(squareError, 0.0f, 60000.0f)));
}

kernel void pyrowave_rate_control_tile_stats_bucket_indices(
    device const RateControlQuantStats *tileStats [[buffer(0)]],
    device const uint *packetByteCosts [[buffer(1)]],
    device uint *bucketIndices [[buffer(2)]],
    constant RateControlBucketConstants &constants [[buffer(3)]],
    uint blockIndex [[thread_position_in_grid]]
) {
    if (blockIndex >= constants.blockCount) {
        return;
    }

    constexpr uint candidateCount = 15u;
    constexpr uint tileCount = 16u;
    constexpr uint bucketCount = 128u;
    constexpr uint bucketClusterWidth = 16u;
    uint outputOffset = blockIndex * candidateCount;
    uint tileStatsOffset = blockIndex * tileCount * candidateCount;
    float distortions[candidateCount];

    for (uint quantLevel = 0u; quantLevel < candidateCount; ++quantLevel) {
        float distortion = 0.0f;
        for (uint tile = 0u; tile < tileCount; ++tile) {
            distortion += pyrowave_quantized_square_error(tileStats[tileStatsOffset + tile * candidateCount + quantLevel].squareError);
        }
        distortions[quantLevel] = distortion;
    }

    float baseDistortion = distortions[0];
    uint baseCost = packetByteCosts[outputOffset];
    uint raw[candidateCount];

    for (uint quantLevel = 0u; quantLevel < candidateCount; ++quantLevel) {
        uint bucket = 0u;
        if (quantLevel != 0u && packetByteCosts[outputOffset + quantLevel] != baseCost) {
            float distortionDelta = max(distortions[quantLevel] - baseDistortion, 0.0f);
            int saving = int(baseCost) - int(packetByteCosts[outputOffset + quantLevel]);
            float costSaving = saving > 0 ? float(saving) : 1.40129846e-45f;
            float index = 60.0f + 2.0f * log2(distortionDelta / costSaving);
            if (isfinite(index)) {
                bucket = uint(clamp(floor(index + 0.5f), 0.0f, 127.0f));
            }
        }
        raw[quantLevel] = min(bucket, bucketCount - bucketClusterWidth + quantLevel);
    }

    for (uint quantLevel = 0u; quantLevel < candidateCount; ++quantLevel) {
        uint bucket = raw[quantLevel];
        for (uint previous = 0u; previous < quantLevel; ++previous) {
            bucket = max(bucket, raw[previous] + quantLevel - previous);
        }
        bucketIndices[outputOffset + quantLevel] = min(bucketCount - 1u, bucket);
    }
}

kernel void pyrowave_rate_control_bucket_savings(
    device const uint *bucketIndices [[buffer(0)]],
    device const uint *packetByteCosts [[buffer(1)]],
    device atomic_uint *bucketSavings [[buffer(2)]],
    constant RateControlBucketSavingsConstants &constants [[buffer(3)]],
    uint blockIndex [[thread_position_in_grid]]
) {
    if (blockIndex >= constants.blockCount) {
        return;
    }

    constexpr uint candidateCount = 15u;
    uint offset = blockIndex * candidateCount;
    for (uint quantLevel = 1u; quantLevel < candidateCount; ++quantLevel) {
        uint previousCost = packetByteCosts[offset + quantLevel - 1u];
        uint currentCost = packetByteCosts[offset + quantLevel];
        if (previousCost > currentCost) {
            uint bucket = bucketIndices[offset + quantLevel];
            if (bucket < 128u) {
                atomic_fetch_add_explicit(&bucketSavings[bucket], previousCost - currentCost, memory_order_relaxed);
            }
        }
    }
}

kernel void pyrowave_rate_control_bucket_savings_prefix(
    device const uint *bucketSavings [[buffer(0)]],
    device uint *cumulativeSavings [[buffer(1)]],
    uint bucket [[thread_position_in_grid]]
) {
    if (bucket >= 128u) {
        return;
    }

    uint running = 0u;
    for (uint index = 0u; index <= bucket; ++index) {
        running += bucketSavings[index];
    }
    cumulativeSavings[bucket] = running;
}

kernel void pyrowave_dwt_lift_rows(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (x & 1u) != 0u;
    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 0: active = odd; factor = dwtAlpha; break;
    case 1: active = !odd; factor = dwtBeta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtDelta; break;
    default: break;
    }

    uint index = y * constants.stride + x;
    if (constants.phase < 4) {
        if (!active) {
            return;
        }
        int left = mirrorIndex(int(x) - 1, int(constants.activeWidth));
        int right = mirrorIndex(int(x) + 1, int(constants.activeWidth));
        samples[index] += factor * (samples[y * constants.stride + uint(left)] + samples[y * constants.stride + uint(right)]);
    } else {
        samples[index] *= odd ? dwtK : dwtInvK;
    }
}

kernel void pyrowave_dwt_lift_columns(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (y & 1u) != 0u;
    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 0: active = odd; factor = dwtAlpha; break;
    case 1: active = !odd; factor = dwtBeta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtDelta; break;
    default: break;
    }

    uint index = y * constants.stride + x;
    if (constants.phase < 4) {
        if (!active) {
            return;
        }
        int top = mirrorIndex(int(y) - 1, int(constants.activeHeight));
        int bottom = mirrorIndex(int(y) + 1, int(constants.activeHeight));
        samples[index] += factor * (samples[uint(top) * constants.stride + x] + samples[uint(bottom) * constants.stride + x]);
    } else {
        samples[index] *= odd ? dwtK : dwtInvK;
    }
}

kernel void pyrowave_dwt_pack_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeWidth + 1u) >> 1u;
    uint packedX = ((x & 1u) == 0u) ? (x >> 1u) : (lowCount + (x >> 1u));
    output[y * constants.stride + packedX] = input[y * constants.stride + x];
}

kernel void pyrowave_dwt_pack_columns(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeHeight + 1u) >> 1u;
    uint packedY = ((y & 1u) == 0u) ? (y >> 1u) : (lowCount + (y >> 1u));
    output[packedY * constants.stride + x] = input[y * constants.stride + x];
}

kernel void pyrowave_dwt_unpack_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeWidth + 1u) >> 1u;
    uint packedX = ((x & 1u) == 0u) ? (x >> 1u) : (lowCount + (x >> 1u));
    output[y * constants.stride + x] = input[y * constants.stride + packedX];
}

kernel void pyrowave_dwt_unpack_columns(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeHeight + 1u) >> 1u;
    uint packedY = ((y & 1u) == 0u) ? (y >> 1u) : (lowCount + (y >> 1u));
    output[y * constants.stride + x] = input[packedY * constants.stride + x];
}

kernel void pyrowave_idwt_unpack_rows_scaled(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeWidth + 1u) >> 1u;
    uint packedX = ((x & 1u) == 0u) ? (x >> 1u) : (lowCount + (x >> 1u));
    float scale = ((x & 1u) != 0u) ? dwtInvK : dwtK;
    output[y * constants.stride + x] = input[y * constants.stride + packedX] * scale;
}

kernel void pyrowave_idwt_unpack_columns_scaled(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeHeight + 1u) >> 1u;
    uint packedY = ((y & 1u) == 0u) ? (y >> 1u) : (lowCount + (y >> 1u));
    float scale = ((y & 1u) != 0u) ? dwtInvK : dwtK;
    output[y * constants.stride + x] = input[packedY * constants.stride + x] * scale;
}

kernel void pyrowave_idwt_lift_rows(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (x & 1u) != 0u;
    uint index = y * constants.stride + x;
    if (constants.phase == 0) {
        samples[index] *= odd ? dwtInvK : dwtK;
        return;
    }

    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 1: active = !odd; factor = dwtDelta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtBeta; break;
    case 4: active = odd; factor = dwtAlpha; break;
    default: break;
    }

    if (!active) {
        return;
    }
    int left = mirrorIndex(int(x) - 1, int(constants.activeWidth));
    int right = mirrorIndex(int(x) + 1, int(constants.activeWidth));
    samples[index] -= factor * (samples[y * constants.stride + uint(left)] + samples[y * constants.stride + uint(right)]);
}

kernel void pyrowave_idwt_lift_columns(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (y & 1u) != 0u;
    uint index = y * constants.stride + x;
    if (constants.phase == 0) {
        samples[index] *= odd ? dwtInvK : dwtK;
        return;
    }

    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 1: active = !odd; factor = dwtDelta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtBeta; break;
    case 4: active = odd; factor = dwtAlpha; break;
    default: break;
    }

    if (!active) {
        return;
    }
    int top = mirrorIndex(int(y) - 1, int(constants.activeHeight));
    int bottom = mirrorIndex(int(y) + 1, int(constants.activeHeight));
    samples[index] -= factor * (samples[uint(top) * constants.stride + x] + samples[uint(bottom) * constants.stride + x]);
}
