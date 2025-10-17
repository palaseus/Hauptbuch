//! Performance benchmarks for the PoS consensus system

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hauptbuch::consensus::pos::{Block, BlockProposal, PoSConsensus, Validator};
use std::time::{SystemTime, UNIX_EPOCH};

fn create_benchmark_validator(id: &str, stake: u64) -> Validator {
    Validator {
        id: id.to_string(),
        stake,
        public_key: vec![id.as_bytes()[0]; 64],
        quantum_public_key: None,
        is_active: true,
        blocks_proposed: 0,
        slash_count: 0,
    }
}

fn create_benchmark_block(height: u64, proposer_id: &str, previous_hash: Vec<u8>) -> Block {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Block {
        hash: vec![height as u8; 32],
        previous_hash,
        height,
        timestamp,
        proposer_id: proposer_id.to_string(),
        merkle_root: vec![0u8; 32],
        nonce: 12345,
        pow_hash: vec![0u8; 32],
        vdf_output: vec![0u8; 32],
        signature: vec![0u8; 64],
        quantum_signature: None,
    }
}

fn benchmark_validator_selection(c: &mut Criterion) {
    let mut consensus = PoSConsensus::new();

    // Add 100 validators
    for i in 0..100 {
        consensus
            .add_validator(create_benchmark_validator(
                &format!("validator{}", i),
                1000 + i as u64,
            ))
            .unwrap();
    }

    c.bench_function("validator_selection", |b| {
        b.iter(|| {
            let seed = black_box(b"benchmark_seed");
            consensus.select_validator(seed)
        })
    });
}

fn benchmark_vdf_calculation(c: &mut Criterion) {
    let consensus = PoSConsensus::new();

    c.bench_function("vdf_calculation", |b| {
        b.iter(|| {
            let input = black_box(b"vdf_input_data");
            consensus.calculate_vdf(input)
        })
    });
}

fn benchmark_block_validation(c: &mut Criterion) {
    let mut consensus = PoSConsensus::new();
    consensus
        .add_validator(create_benchmark_validator("validator1", 1000))
        .unwrap();

    let block = create_benchmark_block(0, "validator1", vec![0u8; 32]);
    let proposal = BlockProposal {
        block,
        vdf_proof: vec![0u8; 32],
        pow_proof: vec![0u8; 32],
        quantum_proof: None,
    };

    c.bench_function("block_validation", |b| {
        b.iter(|| consensus.validate_proposal(black_box(&proposal)))
    });
}

fn benchmark_sha3_hashing(c: &mut Criterion) {
    use sha3::{Digest, Sha3_256};

    c.bench_function("sha3_hashing", |b| {
        b.iter(|| {
            let data = black_box(b"test_data_for_hashing");
            let mut hasher = Sha3_256::new();
            hasher.update(data);
            hasher.finalize().to_vec()
        })
    });
}

fn benchmark_pow_verification(c: &mut Criterion) {
    c.bench_function("pow_verification", |b| {
        b.iter(|| {
            let hash = [0u8; 32];
            let difficulty = black_box(4);
            // Use a simple difficulty check for benchmarking
            hash.iter().take((difficulty / 8) as usize).all(|&b| b == 0)
        })
    });
}

criterion_group!(
    benches,
    benchmark_validator_selection,
    benchmark_vdf_calculation,
    benchmark_block_validation,
    benchmark_sha3_hashing,
    benchmark_pow_verification
);
criterion_main!(benches);
