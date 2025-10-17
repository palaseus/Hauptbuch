//! Python FFI Bindings for Quantum-Resistant Cryptography
//!
//! This module provides Python bindings for the quantum-resistant cryptographic
//! operations using PyO3.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

use crate::crypto::{
    CryptoConfig,
    CryptoEngine, 
    MLKemKeypair, 
    MLDsaKeypair, 
    SLHDsaKeypair,
    Signature,
    VerificationResult
};

/// Python module for Hauptbuch crypto operations
#[pymodule]
fn hauptbuch_crypto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PythonCryptoEngine>()?;
    m.add_class::<MLKemKeypair>()?;
    m.add_class::<MLDsaKeypair>()?;
    m.add_class::<SLHDsaKeypair>()?;
    m.add_class::<Signature>()?;
    m.add_class::<VerificationResult>()?;
    Ok(())
}

/// Python wrapper for the crypto engine
#[pyclass]
pub struct PythonCryptoEngine {
    engine: CryptoEngine,
}

#[pymethods]
impl PythonCryptoEngine {
    #[new]
    fn new() -> PyResult<Self> {
        let config = CryptoConfig {
            quantum_resistant: true,
            hybrid_mode: true,
            nist_pqc_enabled: true,
            ml_kem_enabled: true,
            ml_dsa_enabled: true,
            slh_dsa_enabled: true,
        };
        
        let engine = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(CryptoEngine::new(config))
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize crypto engine: {}", e)))?;
        
        Ok(Self { engine })
    }
    
    /// Generate ML-KEM keypair
    fn generate_ml_kem_keypair(&self, key_size: usize) -> PyResult<MLKemKeypair> {
        let keypair = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.generate_ml_kem_keypair(key_size))
            .map_err(|e| PyValueError::new_err(format!("Failed to generate ML-KEM keypair: {}", e)))?;
        
        Ok(keypair)
    }
    
    /// Generate ML-DSA keypair
    fn generate_ml_dsa_keypair(&self, key_size: usize) -> PyResult<MLDsaKeypair> {
        let keypair = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.generate_ml_dsa_keypair(key_size))
            .map_err(|e| PyValueError::new_err(format!("Failed to generate ML-DSA keypair: {}", e)))?;
        
        Ok(keypair)
    }
    
    /// Generate SLH-DSA keypair
    fn generate_slh_dsa_keypair(&self, key_size: usize) -> PyResult<SLHDsaKeypair> {
        let keypair = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.generate_slh_dsa_keypair(key_size))
            .map_err(|e| PyValueError::new_err(format!("Failed to generate SLH-DSA keypair: {}", e)))?;
        
        Ok(keypair)
    }
    
    /// Sign message with ML-DSA
    fn sign_ml_dsa(&self, message: &[u8], private_key: &[u8]) -> PyResult<Signature> {
        let signature = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.sign_ml_dsa(message, private_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to sign message: {}", e)))?;
        
        Ok(signature)
    }
    
    /// Sign message with SLH-DSA
    fn sign_slh_dsa(&self, message: &[u8], private_key: &[u8]) -> PyResult<Signature> {
        let signature = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.sign_slh_dsa(message, private_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to sign message: {}", e)))?;
        
        Ok(signature)
    }
    
    /// Verify ML-DSA signature
    fn verify_ml_dsa(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> PyResult<VerificationResult> {
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.verify_ml_dsa(message, signature, public_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to verify signature: {}", e)))?;
        
        Ok(result)
    }
    
    /// Verify SLH-DSA signature
    fn verify_slh_dsa(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> PyResult<VerificationResult> {
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.verify_slh_dsa(message, signature, public_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to verify signature: {}", e)))?;
        
        Ok(result)
    }
    
    /// Encrypt with ML-KEM
    fn encrypt_ml_kem(&self, plaintext: &[u8], public_key: &[u8]) -> PyResult<Vec<u8>> {
        let ciphertext = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.encrypt_ml_kem(plaintext, public_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to encrypt: {}", e)))?;
        
        Ok(ciphertext)
    }
    
    /// Decrypt with ML-KEM
    fn decrypt_ml_kem(&self, ciphertext: &[u8], private_key: &[u8]) -> PyResult<Vec<u8>> {
        let plaintext = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.decrypt_ml_kem(ciphertext, private_key))
            .map_err(|e| PyValueError::new_err(format!("Failed to decrypt: {}", e)))?;
        
        Ok(plaintext)
    }
    
    /// Generate hybrid keypair (quantum-resistant + classical)
    fn generate_hybrid_keypair(&self, quantum_algorithm: &str, classical_algorithm: &str) -> PyResult<HashMap<String, Vec<u8>>> {
        let keypair = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.generate_hybrid_keypair(quantum_algorithm, classical_algorithm))
            .map_err(|e| PyValueError::new_err(format!("Failed to generate hybrid keypair: {}", e)))?;
        
        let mut result = HashMap::new();
        result.insert("quantum_private".to_string(), keypair.quantum_private_key);
        result.insert("quantum_public".to_string(), keypair.quantum_public_key);
        result.insert("classical_private".to_string(), keypair.classical_private_key);
        result.insert("classical_public".to_string(), keypair.classical_public_key);
        result.insert("address".to_string(), keypair.address);
        
        Ok(result)
    }
    
    /// Sign with hybrid cryptography
    fn sign_hybrid(&self, message: &[u8], quantum_private: &[u8], classical_private: &[u8]) -> PyResult<HashMap<String, Vec<u8>>> {
        let signature = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.sign_hybrid(message, quantum_private, classical_private))
            .map_err(|e| PyValueError::new_err(format!("Failed to sign with hybrid crypto: {}", e)))?;
        
        let mut result = HashMap::new();
        result.insert("quantum_signature".to_string(), signature.quantum_signature);
        result.insert("classical_signature".to_string(), signature.classical_signature);
        result.insert("combined_signature".to_string(), signature.combined_signature);
        
        Ok(result)
    }
    
    /// Verify hybrid signature
    fn verify_hybrid(&self, message: &[u8], signature: &HashMap<String, Vec<u8>>, quantum_public: &[u8], classical_public: &[u8]) -> PyResult<bool> {
        let quantum_signature = signature.get("quantum_signature")
            .ok_or_else(|| PyValueError::new_err("Missing quantum signature"))?;
        let classical_signature = signature.get("classical_signature")
            .ok_or_else(|| PyValueError::new_err("Missing classical signature"))?;
        
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.engine.verify_hybrid(message, quantum_signature, classical_signature, quantum_public, classical_public))
            .map_err(|e| PyValueError::new_err(format!("Failed to verify hybrid signature: {}", e)))?;
        
        Ok(result)
    }
}

/// Python wrapper for ML-KEM keypair
#[pyclass]
pub struct MLKemKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[pymethods]
impl MLKemKeypair {
    #[getter]
    fn private_key(&self) -> &[u8] {
        &self.private_key
    }
    
    #[getter]
    fn public_key(&self) -> &[u8] {
        &self.public_key
    }
    
    #[getter]
    fn address(&self) -> &[u8] {
        &self.address
    }
    
    fn to_hex(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();
        result.insert("private_key".to_string(), hex::encode(&self.private_key));
        result.insert("public_key".to_string(), hex::encode(&self.public_key));
        result.insert("address".to_string(), hex::encode(&self.address));
        result
    }
}

/// Python wrapper for ML-DSA keypair
#[pyclass]
pub struct MLDsaKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[pymethods]
impl MLDsaKeypair {
    #[getter]
    fn private_key(&self) -> &[u8] {
        &self.private_key
    }
    
    #[getter]
    fn public_key(&self) -> &[u8] {
        &self.public_key
    }
    
    #[getter]
    fn address(&self) -> &[u8] {
        &self.address
    }
    
    fn to_hex(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();
        result.insert("private_key".to_string(), hex::encode(&self.private_key));
        result.insert("public_key".to_string(), hex::encode(&self.public_key));
        result.insert("address".to_string(), hex::encode(&self.address));
        result
    }
}

/// Python wrapper for SLH-DSA keypair
#[pyclass]
pub struct SLHDsaKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[pymethods]
impl SLHDsaKeypair {
    #[getter]
    fn private_key(&self) -> &[u8] {
        &self.private_key
    }
    
    #[getter]
    fn public_key(&self) -> &[u8] {
        &self.public_key
    }
    
    #[getter]
    fn address(&self) -> &[u8] {
        &self.address
    }
    
    fn to_hex(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();
        result.insert("private_key".to_string(), hex::encode(&self.private_key));
        result.insert("public_key".to_string(), hex::encode(&self.public_key));
        result.insert("address".to_string(), hex::encode(&self.address));
        result
    }
}

/// Python wrapper for signature
#[pyclass]
pub struct Signature {
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub algorithm: String,
}

#[pymethods]
impl Signature {
    #[getter]
    fn signature(&self) -> &[u8] {
        &self.signature
    }
    
    #[getter]
    fn public_key(&self) -> &[u8] {
        &self.public_key
    }
    
    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }
    
    fn to_hex(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();
        result.insert("signature".to_string(), hex::encode(&self.signature));
        result.insert("public_key".to_string(), hex::encode(&self.public_key));
        result.insert("algorithm".to_string(), self.algorithm.clone());
        result
    }
}

/// Python wrapper for verification result
#[pyclass]
pub struct VerificationResult {
    pub valid: bool,
    pub algorithm: String,
    pub details: Option<String>,
}

#[pymethods]
impl VerificationResult {
    #[getter]
    fn valid(&self) -> bool {
        self.valid
    }
    
    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }
    
    #[getter]
    fn details(&self) -> Option<&str> {
        self.details.as_deref()
    }
    
    fn to_dict(&self) -> HashMap<String, PyObject> {
        let mut result = HashMap::new();
        result.insert("valid".to_string(), Python::with_gil(|py| self.valid.into_py(py)));
        result.insert("algorithm".to_string(), Python::with_gil(|py| self.algorithm.clone().into_py(py)));
        result.insert("details".to_string(), Python::with_gil(|py| self.details.clone().into_py(py)));
        result
    }
}

/// Utility functions for Python integration
#[pyfunction]
fn create_crypto_engine() -> PyResult<PythonCryptoEngine> {
    PythonCryptoEngine::new()
}

#[pyfunction]
fn hex_to_bytes(hex_string: &str) -> PyResult<Vec<u8>> {
    hex::decode(hex_string)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex string: {}", e)))
}

#[pyfunction]
fn bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Python module initialization
pub fn init_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_crypto_engine, m)?)?;
    m.add_function(wrap_pyfunction!(hex_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(bytes_to_hex, m)?)?;
    Ok(())
}
