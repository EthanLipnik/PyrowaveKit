import Foundation
import Metal

extension Plane8 {
    public init(texture: MTLTexture) throws {
        guard texture.pixelFormat == .r8Unorm,
              texture.width > 0,
              texture.height > 0 else {
            throw PyrowaveError.unsupportedFormat("Metal texture import expects r8Unorm planes")
        }

        var data = Array(repeating: UInt8(0), count: texture.width * texture.height)
        data.withUnsafeMutableBytes { destination in
            texture.getBytes(
                destination.baseAddress!,
                bytesPerRow: texture.width,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                mipmapLevel: 0
            )
        }
        try self.init(width: texture.width, height: texture.height, data: data)
    }

    public func makeMetalTexture(device: MTLDevice, usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw PyrowaveError.processFailed("failed to allocate Metal texture")
        }
        try copy(to: texture)
        return texture
    }

    public func copy(to texture: MTLTexture) throws {
        guard texture.pixelFormat == .r8Unorm,
              texture.width == width,
              texture.height == height else {
            throw PyrowaveError.invalidDimensions
        }

        data.withUnsafeBytes { source in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: source.baseAddress!,
                bytesPerRow: width
            )
        }
    }
}

extension YUVFrame {
    public init(
        yTexture: MTLTexture,
        cbTexture: MTLTexture,
        crTexture: MTLTexture,
        videoSignal: VideoSignalMetadata = .default
    ) throws {
        let y = try Plane8(texture: yTexture)
        let cb = try Plane8(texture: cbTexture)
        let cr = try Plane8(texture: crTexture)
        let chroma: ChromaSubsampling
        if cb.width == y.width / 2,
           cb.height == y.height / 2,
           cr.width == cb.width,
           cr.height == cb.height {
            chroma = .yuv420
        } else if cb.width == y.width,
                  cb.height == y.height,
                  cr.width == y.width,
                  cr.height == y.height {
            chroma = .yuv444
        } else {
            throw PyrowaveError.invalidDimensions
        }

        try self.init(
            width: y.width,
            height: y.height,
            chroma: chroma,
            y: y,
            cb: cb,
            cr: cr,
            videoSignal: videoSignal
        )
    }

    public func makeMetalTextures(
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) throws -> (y: MTLTexture, cb: MTLTexture, cr: MTLTexture) {
        (
            y: try y.makeMetalTexture(device: device, usage: usage),
            cb: try cb.makeMetalTexture(device: device, usage: usage),
            cr: try cr.makeMetalTexture(device: device, usage: usage)
        )
    }

    public func copy(
        toYTexture yTexture: MTLTexture,
        cbTexture: MTLTexture,
        crTexture: MTLTexture
    ) throws {
        try y.copy(to: yTexture)
        try cb.copy(to: cbTexture)
        try cr.copy(to: crTexture)
    }
}

extension PyrowaveCodec {
    public func decodeToMetalTextures(
        _ frame: EncodedFrame,
        device: MTLDevice? = MTLCreateSystemDefaultDevice(),
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) throws -> (y: MTLTexture, cb: MTLTexture, cr: MTLTexture) {
        guard let device else {
            throw PyrowaveError.externalToolUnavailable("Metal device")
        }
        return try decode(frame).makeMetalTextures(device: device, usage: usage)
    }
}

extension PyrowavePacketStreamDecoder {
    public func decodeToMetalTextures(
        allowPartialFrame: Bool = false,
        device: MTLDevice? = MTLCreateSystemDefaultDevice(),
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) throws -> (y: MTLTexture, cb: MTLTexture, cr: MTLTexture) {
        guard let device else {
            throw PyrowaveError.externalToolUnavailable("Metal device")
        }
        return try decode(allowPartialFrame: allowPartialFrame).makeMetalTextures(device: device, usage: usage)
    }
}
