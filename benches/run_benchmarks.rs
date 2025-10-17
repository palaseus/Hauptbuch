//! Performance Benchmark Runner
//! 
//! This script runs the performance benchmarking suite and outputs
//! comprehensive results for the decentralized voting blockchain.

use hauptbuch::benchmarks::{
    PerformanceBenchmark, BenchmarkConfig, OutputFormat
};

fn main() {
    println!("🚀 Starting Decentralized Voting Blockchain Performance Benchmarks");
    println!("{}", "=".repeat(80));
    
    // Run benchmarks with different configurations
    let configs = vec![
        ("Low Load", BenchmarkConfig {
            node_count: 10,
            shard_count: 2,
            transaction_count: 100,
            duration_seconds: 10,
            enable_stress_tests: false,
            enable_resource_monitoring: true,
            output_format: OutputFormat::Human,
        }),
        ("Medium Load", BenchmarkConfig {
            node_count: 50,
            shard_count: 10,
            transaction_count: 1000,
            duration_seconds: 30,
            enable_stress_tests: true,
            enable_resource_monitoring: true,
            output_format: OutputFormat::Both,
        }),
        ("High Load", BenchmarkConfig {
            node_count: 100,
            shard_count: 20,
            transaction_count: 10000,
            duration_seconds: 60,
            enable_stress_tests: true,
            enable_resource_monitoring: true,
            output_format: OutputFormat::Json,
        }),
    ];
    
    for (name, config) in configs {
        println!("\n📊 Running {} Benchmarks", name);
        println!("{}", "-".repeat(50));
        
        let mut benchmark = PerformanceBenchmark::new(config);
        let results = benchmark.run_benchmarks();
        
        println!("\n📈 {} Results Summary:", name);
        println!("  Throughput: {:.2} tx/sec", results.throughput.transactions_per_second);
        println!("  Votes/sec: {:.2}", results.throughput.votes_per_second);
        println!("  Transfers/sec: {:.2}", results.throughput.transfers_per_second);
        println!("  Avg Latency: {:.2}ms", results.latency.average_latency_ms);
        println!("  CPU Usage: {:.1}%", results.resources.cpu_usage_percent);
        println!("  Memory Usage: {:.2}MB", results.resources.memory_usage_mb);
        println!("  Performance Score: {:.1}/100", results.summary.performance_score);
        println!("  Success Rate: {:.1}%", results.summary.success_rate);
    }
    
    println!("\n🎯 Benchmark Suite Complete!");
    println!("{}", "=".repeat(80));
}
