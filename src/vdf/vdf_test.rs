//! Comprehensive test suite for the Enhanced VDF Module
//!
//! This module contains extensive tests for the VDF implementation, covering
//! normal operation, edge cases, malicious behavior, and stress tests to ensure
//! robustness and security of the VDF system.

use crate::vdf::engine::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a test VDF engine
    fn create_test_vdf_engine() -> VDFEngine {
        VDFEngine::with_params(
            0xFFFFFFFFFFFFFFC5, // Test modulus
            2,                  // Generator
            100,                // Reduced iterations for testing
            32,                 // Security parameter
        )
    }

    /// Helper function to create test input data
    fn create_test_input(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    // ===== NORMAL OPERATION TESTS =====

    #[test]
    fn test_vdf_creation() {
        let vdf = VDFEngine::new();
        assert_eq!(vdf.get_params().modulus, 0xFFFFFFFFFFFFFFC5);
        assert_eq!(vdf.get_params().generator, 2);
        assert_eq!(vdf.get_params().iterations, 1000);
        assert_eq!(vdf.get_params().security_param, 128);
        println!("✅ VDF creation test passed");
    }

    #[test]
    fn test_vdf_custom_params() {
        let vdf = VDFEngine::with_params(1000, 3, 500, 64);
        assert_eq!(vdf.get_params().modulus, 1000);
        assert_eq!(vdf.get_params().generator, 3);
        assert_eq!(vdf.get_params().iterations, 500);
        assert_eq!(vdf.get_params().security_param, 64);
        println!("✅ VDF custom parameters test passed");
    }

    #[test]
    fn test_vdf_evaluation() {
        let mut vdf = create_test_vdf_engine();
        let input = b"test_input";
        let output = vdf.evaluate(input);

        assert!(!output.is_empty(), "VDF output should not be empty");
        assert_eq!(output.len(), 8, "VDF output should be 8 bytes");

        // VDF should be deterministic
        let output2 = vdf.evaluate(input);
        assert_eq!(output, output2, "VDF should be deterministic");

        println!("✅ VDF evaluation test passed");
    }

    #[test]
    fn test_vdf_proof_generation() {
        let mut vdf = create_test_vdf_engine();
        let input = b"proof_test";
        let output = vdf.evaluate(input);
        let proof = vdf.generate_proof(input, &output);

        assert_eq!(proof.input, input);
        assert_eq!(proof.output, output);
        assert_eq!(proof.iterations, vdf.get_params().iterations);
        assert!(
            !proof.proof_elements.is_empty(),
            "Proof elements should not be empty"
        );
        assert!(
            !proof.signature.is_empty(),
            "Proof signature should not be empty"
        );
        assert!(proof.timestamp > 0, "Proof timestamp should be positive");

        println!("✅ VDF proof generation test passed");
    }

    #[test]
    fn test_vdf_proof_verification() {
        let mut vdf = create_test_vdf_engine();
        let input = b"verification_test";
        let output = vdf.evaluate(input);
        let proof = vdf.generate_proof(input, &output);

        let is_valid = vdf.verify_proof(&proof);
        assert!(is_valid, "Valid proof should be verified as true");

        println!("✅ VDF proof verification test passed");
    }

    #[test]
    fn test_vdf_caching() {
        let mut vdf = create_test_vdf_engine();
        let input = b"cache_test";

        // First evaluation should populate cache
        let _output1 = vdf.evaluate(input);
        let (precomputed_size, _) = vdf.get_cache_stats();
        assert!(
            precomputed_size > 0,
            "Cache should contain entries after evaluation"
        );

        // Second evaluation should use cache
        let _output2 = vdf.evaluate(input);

        // Clear cache and verify it's empty
        vdf.clear_caches();
        let (precomputed_size, verification_size) = vdf.get_cache_stats();
        assert_eq!(
            precomputed_size, 0,
            "Precomputed cache should be empty after clearing"
        );
        assert_eq!(
            verification_size, 0,
            "Verification cache should be empty after clearing"
        );

        println!("✅ VDF caching test passed");
    }

    #[test]
    fn test_vdf_parameter_update() {
        let mut vdf = create_test_vdf_engine();
        let original_iterations = vdf.get_params().iterations;

        let new_params = VDFParams {
            modulus: 0xFFFFFFFFFFFFFFC5,
            generator: 2,
            iterations: 200,
            security_param: 64,
        };

        vdf.update_params(new_params);
        assert_eq!(vdf.get_params().iterations, 200);
        assert_ne!(vdf.get_params().iterations, original_iterations);

        println!("✅ VDF parameter update test passed");
    }

    // ===== EDGE CASE TESTS =====

    #[test]
    fn test_vdf_empty_input() {
        let mut vdf = create_test_vdf_engine();
        let input = b"";
        let output = vdf.evaluate(input);

        assert!(!output.is_empty(), "VDF should handle empty input");
        assert_eq!(
            output.len(),
            8,
            "VDF output should be 8 bytes even for empty input"
        );

        println!("✅ VDF empty input test passed");
    }

    #[test]
    fn test_vdf_large_input() {
        let mut vdf = create_test_vdf_engine();
        let input = create_test_input(10000); // 10KB input
        let output = vdf.evaluate(&input);

        assert!(!output.is_empty(), "VDF should handle large input");
        assert_eq!(output.len(), 8, "VDF output should be 8 bytes");

        println!("✅ VDF large input test passed");
    }

    #[test]
    fn test_vdf_zero_input() {
        let mut vdf = create_test_vdf_engine();
        let input = vec![0u8; 32];
        let output = vdf.evaluate(&input);

        assert!(!output.is_empty(), "VDF should handle zero input");
        assert_eq!(output.len(), 8, "VDF output should be 8 bytes");

        println!("✅ VDF zero input test passed");
    }

    #[test]
    fn test_vdf_extreme_iterations() {
        let mut vdf = VDFEngine::with_params(
            0xFFFFFFFFFFFFFFC5,
            2,
            1, // Minimal iterations
            32,
        );

        let input = b"extreme_test";
        let output = vdf.evaluate(input);

        assert!(
            !output.is_empty(),
            "VDF should handle extreme iteration counts"
        );

        println!("✅ VDF extreme iterations test passed");
    }

    #[test]
    fn test_vdf_invalid_proof_elements() {
        let mut vdf = create_test_vdf_engine();
        let input = b"invalid_proof_test";
        let output = vdf.evaluate(input);

        // Create proof with invalid elements
        let mut proof = vdf.generate_proof(input, &output);
        proof.proof_elements.clear(); // Remove proof elements

        let is_valid = vdf.verify_proof(&proof);
        assert!(!is_valid, "Proof with invalid elements should be rejected");

        println!("✅ VDF invalid proof elements test passed");
    }

    #[test]
    fn test_vdf_mismatched_input_output() {
        let mut vdf = create_test_vdf_engine();
        let input1 = b"input1";
        let input2 = b"input2";
        let _output1 = vdf.evaluate(input1);
        let output2 = vdf.evaluate(input2);

        // Create proof with mismatched input/output
        let proof = vdf.generate_proof(input1, &output2);
        let is_valid = vdf.verify_proof(&proof);
        assert!(
            !is_valid,
            "Proof with mismatched input/output should be rejected"
        );

        println!("✅ VDF mismatched input/output test passed");
    }

    // ===== MALICIOUS BEHAVIOR TESTS =====

    #[test]
    fn test_vdf_forged_proof() {
        let mut vdf = create_test_vdf_engine();
        let input = b"forged_proof_test";
        let output = vdf.evaluate(input);

        // Create forged proof with different output
        let mut forged_output = output.clone();
        forged_output[0] = forged_output[0].wrapping_add(1); // Modify output

        let proof = vdf.generate_proof(input, &forged_output);
        let is_valid = vdf.verify_proof(&proof);
        assert!(!is_valid, "Forged proof should be rejected");

        println!("✅ VDF forged proof test passed");
    }

    #[test]
    fn test_vdf_manipulated_randomness() {
        let mut vdf = create_test_vdf_engine();
        let input = b"manipulation_test";

        // Test that VDF output is unpredictable
        let output1 = vdf.evaluate(input);
        let output2 = vdf.evaluate(input);

        // VDF should be deterministic, not random
        assert_eq!(output1, output2, "VDF should be deterministic");

        // Test with different inputs
        let input2 = b"different_input";
        let output3 = vdf.evaluate(input2);
        assert_ne!(
            output1, output3,
            "Different inputs should produce different outputs"
        );

        println!("✅ VDF manipulated randomness test passed");
    }

    #[test]
    fn test_vdf_invalid_signature() {
        let mut vdf = create_test_vdf_engine();
        let input = b"signature_test";
        let output = vdf.evaluate(input);
        let mut proof = vdf.generate_proof(input, &output);

        // Corrupt the signature
        proof.signature[0] = proof.signature[0].wrapping_add(1);

        let is_valid = vdf.verify_proof(&proof);
        assert!(!is_valid, "Proof with invalid signature should be rejected");

        println!("✅ VDF invalid signature test passed");
    }

    #[test]
    fn test_vdf_replay_attack() {
        let mut vdf = create_test_vdf_engine();
        let input = b"replay_attack_test";
        let output = vdf.evaluate(input);
        let proof = vdf.generate_proof(input, &output);

        // Verify the same proof multiple times
        let is_valid1 = vdf.verify_proof(&proof);
        let is_valid2 = vdf.verify_proof(&proof);
        let is_valid3 = vdf.verify_proof(&proof);

        assert!(is_valid1, "First verification should succeed");
        assert!(is_valid2, "Second verification should succeed");
        assert!(is_valid3, "Third verification should succeed");

        println!("✅ VDF replay attack test passed");
    }

    // ===== STRESS TESTS =====

    #[test]
    fn test_vdf_high_iteration_count() {
        let mut vdf = VDFEngine::with_params(
            0xFFFFFFFFFFFFFFC5,
            2,
            10000, // High iteration count
            64,
        );

        let input = b"stress_test_high_iterations";
        let start = std::time::Instant::now();
        let output = vdf.evaluate(input);
        let duration = start.elapsed();

        assert!(
            !output.is_empty(),
            "VDF should complete high iteration count"
        );
        assert!(
            duration.as_secs() < 10,
            "VDF should complete within reasonable time"
        );

        println!("✅ VDF high iteration count test passed: {:?}", duration);
    }

    #[test]
    fn test_vdf_concurrent_evaluations() {
        let mut vdf = create_test_vdf_engine();

        // Perform multiple evaluations sequentially (simulating concurrent access)
        for i in 0..10 {
            let input = format!("concurrent_test_{}", i).into_bytes();
            let output = vdf.evaluate(&input);
            assert!(
                !output.is_empty(),
                "Concurrent evaluation {} should succeed",
                i
            );
        }

        println!("✅ VDF concurrent evaluations test passed");
    }

    #[test]
    fn test_vdf_memory_stress() {
        let mut vdf = create_test_vdf_engine();

        // Perform many evaluations to stress test memory
        for i in 0..1000 {
            let input = format!("memory_stress_{}", i).into_bytes();
            let _output = vdf.evaluate(&input);
        }

        let (precomputed_size, _verification_size) = vdf.get_cache_stats();
        assert!(
            precomputed_size > 0,
            "Cache should contain entries after stress test"
        );

        // Clear cache to free memory
        vdf.clear_caches();
        let (precomputed_size, verification_size) = vdf.get_cache_stats();
        assert_eq!(precomputed_size, 0, "Cache should be empty after clearing");
        assert_eq!(
            verification_size, 0,
            "Verification cache should be empty after clearing"
        );

        println!("✅ VDF memory stress test passed");
    }

    #[test]
    fn test_vdf_large_proof_generation() {
        let mut vdf = VDFEngine::with_params(
            0xFFFFFFFFFFFFFFC5,
            2,
            1000,
            256, // Large security parameter
        );

        let input = b"large_proof_test";
        let output = vdf.evaluate(input);
        let proof = vdf.generate_proof(input, &output);

        assert_eq!(
            proof.proof_elements.len(),
            256,
            "Proof should have correct number of elements"
        );
        assert!(vdf.verify_proof(&proof), "Large proof should be valid");

        println!("✅ VDF large proof generation test passed");
    }

    // ===== INTEGRATION TESTS =====

    #[test]
    fn test_pos_integration() {
        let mut vdf = create_test_vdf_engine();
        let seed = b"pos_seed";

        let randomness = generate_pos_randomness(&mut vdf, seed);
        assert!(!randomness.is_empty(), "PoS randomness should not be empty");

        // Generate proof for PoS
        let input = seed;
        let output = vdf.evaluate(input);
        let proof = vdf.generate_proof(input, &output);

        let is_valid = verify_pos_proof(&mut vdf, &proof);
        assert!(is_valid, "PoS proof should be valid");

        println!("✅ VDF PoS integration test passed");
    }

    #[test]
    fn test_sharding_integration() {
        let mut vdf = create_test_vdf_engine();
        let shard_id = 5u32;
        let validator_id = "validator_123";

        let randomness = generate_shard_randomness(&mut vdf, shard_id, validator_id);
        assert!(
            !randomness.is_empty(),
            "Shard randomness should not be empty"
        );

        // Generate proof for sharding
        let mut seed = Vec::new();
        seed.extend_from_slice(&shard_id.to_le_bytes());
        seed.extend_from_slice(validator_id.as_bytes());
        let output = vdf.evaluate(&seed);
        let proof = vdf.generate_proof(&seed, &output);

        let is_valid = verify_shard_proof(&mut vdf, &proof);
        assert!(is_valid, "Shard proof should be valid");

        println!("✅ VDF sharding integration test passed");
    }

    #[test]
    fn test_voting_integration() {
        let mut vdf = create_test_vdf_engine();
        let vote_id = b"vote_456";
        let timestamp = 1234567890u64;

        let randomness = generate_vote_randomness(&mut vdf, vote_id, timestamp);
        assert!(
            !randomness.is_empty(),
            "Vote randomness should not be empty"
        );

        // Generate proof for voting
        let mut seed = Vec::new();
        seed.extend_from_slice(vote_id);
        seed.extend_from_slice(&timestamp.to_le_bytes());
        let output = vdf.evaluate(&seed);
        let proof = vdf.generate_proof(&seed, &output);

        let is_valid = verify_vote_proof(&mut vdf, &proof);
        assert!(is_valid, "Vote proof should be valid");

        println!("✅ VDF voting integration test passed");
    }

    // ===== PERFORMANCE TESTS =====

    #[test]
    fn test_vdf_performance_benchmark() {
        let mut vdf = create_test_vdf_engine();
        let iterations = 100;
        let mut total_time = std::time::Duration::new(0, 0);

        for i in 0..iterations {
            let input = format!("perf_test_{}", i).into_bytes();
            let start = std::time::Instant::now();
            let _output = vdf.evaluate(&input);
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations;
        println!(
            "✅ VDF performance benchmark: Average time per evaluation: {:?}",
            avg_time
        );

        // Performance should be reasonable (adjust threshold as needed)
        assert!(
            avg_time.as_millis() < 100,
            "VDF evaluation should be reasonably fast"
        );
    }

    #[test]
    fn test_vdf_proof_performance() {
        let mut vdf = create_test_vdf_engine();
        let input = b"proof_perf_test";
        let output = vdf.evaluate(input);

        let start = std::time::Instant::now();
        let proof = vdf.generate_proof(input, &output);
        let generation_time = start.elapsed();

        let start = std::time::Instant::now();
        let is_valid = vdf.verify_proof(&proof);
        let verification_time = start.elapsed();

        assert!(is_valid, "Generated proof should be valid");
        println!(
            "✅ VDF proof performance: Generation: {:?}, Verification: {:?}",
            generation_time, verification_time
        );

        // Both operations should be reasonably fast
        assert!(
            generation_time.as_millis() < 1000,
            "Proof generation should be reasonably fast"
        );
        assert!(
            verification_time.as_millis() < 1000,
            "Proof verification should be reasonably fast"
        );
    }
}
