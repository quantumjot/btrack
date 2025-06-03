/*
--------------------------------------------------------------------------------
 Name:     btrack
 Purpose:  Metal implementation for GPU-accelerated belief matrix updates
 Authors:  Based on CUDA implementation by Alan R. Lowe (arl) a.lowe@ucl.ac.uk
 License:  See LICENSE.md
 Created:  Metal port for macOS GPU acceleration
--------------------------------------------------------------------------------
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "belief.h"
#include <iostream>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> pipelineState = nil;
static bool metalInitialized = false;

bool initializeMetal() {
    if (metalInitialized) return true;
    
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return false;
        }
        
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return false;
        }
        
        NSError* error = nil;
        
        // FIXED: Better shader loading strategy for library builds
        NSString* shaderPath = nil;
        
        // Try bundle resource first (for app bundles)
        NSBundle* bundle = [NSBundle mainBundle];
        shaderPath = [bundle pathForResource:@"belief" ofType:@"metal"];
        
        // If not found in bundle, try relative to executable (for library builds)
        if (!shaderPath) {
            NSString* executablePath = [[NSBundle mainBundle] executablePath];
            NSString* executableDir = [executablePath stringByDeletingLastPathComponent];
            shaderPath = [executableDir stringByAppendingPathComponent:@"../src/belief.metal"];
        }
        
        // Final fallback: try current working directory
        if (!shaderPath || ![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
            shaderPath = @"./btrack/src/belief.metal";
        }
        
        if (![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
            std::cerr << "Could not find belief.metal shader file at any expected location" << std::endl;
            return false;
        }
        
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath 
                                                            encoding:NSUTF8StringEncoding 
                                                               error:&error];
        if (!shaderSource) {
            std::cerr << "Failed to load shader source: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        
        if (!library) {
            std::cerr << "Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"track_Metal"];
        if (!kernelFunction) {
            std::cerr << "Failed to create Metal kernel function" << std::endl;
            return false;
        }
        
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            std::cerr << "Failed to create Metal pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        metalInitialized = true;
        std::cout << "[INFO] Metal initialized successfully." << std::endl;
        return true;
    }
}

extern "C" {
void cost_METAL(float *belief_matrix, const float *new_positions,
               const float *predicted_positions, const unsigned int N,
               const unsigned int T, const float prob_not_assign,
               const float accuracy) {
    if (!initializeMetal()) {
        std::cerr << "Failed to initialize Metal" << std::endl;
        return;
    }

    @autoreleasepool {
        // Create Metal buffers
        id<MTLBuffer> beliefBuffer = [device newBufferWithBytes:belief_matrix 
                                                         length:N * T * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> newPosBuffer = [device newBufferWithBytes:new_positions
                                                         length:N * 3 * sizeof(float) 
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> predPosBuffer = [device newBufferWithBytes:predicted_positions
                                                          length:T * 6 * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline state and buffers
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:beliefBuffer offset:0 atIndex:0];
        [encoder setBuffer:newPosBuffer offset:0 atIndex:1];
        [encoder setBuffer:predPosBuffer offset:0 atIndex:2];
        [encoder setBytes:&N length:sizeof(unsigned int) atIndex:3];
        [encoder setBytes:&T length:sizeof(unsigned int) atIndex:4];
        [encoder setBytes:&prob_not_assign length:sizeof(float) atIndex:5];
        [encoder setBytes:&accuracy length:sizeof(float) atIndex:6];
        
        // Configure threadgroup sizes (equivalent to CUDA block(16,16))
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize numThreadgroups = MTLSizeMake((N + 15) / 16, (T + 15) / 16, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back to host memory
        memcpy(belief_matrix, [beliefBuffer contents], N * T * sizeof(float));
    }
}

int main_METAL(void) {
    return 0;
}
}
