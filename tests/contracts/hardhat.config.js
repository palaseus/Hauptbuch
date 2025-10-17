require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-ethers");
require("hardhat-gas-reporter");
require("solidity-coverage");

// Hauptbuch network configuration
const HAUPTBUCH_RPC_URL = process.env.HAUPTBUCH_RPC_URL || "http://localhost:8080";
const HAUPTBUCH_CHAIN_ID = process.env.HAUPTBUCH_CHAIN_ID || 1337;
const HAUPTBUCH_PRIVATE_KEY = process.env.HAUPTBUCH_PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000001";

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hauptbuch: {
      url: HAUPTBUCH_RPC_URL,
      chainId: parseInt(HAUPTBUCH_CHAIN_ID),
      accounts: [HAUPTBUCH_PRIVATE_KEY],
      gas: 8000000,
      gasPrice: 20000000000,
      timeout: 60000
    },
    hauptbuch_testnet: {
      url: HAUPTBUCH_RPC_URL,
      chainId: parseInt(HAUPTBUCH_CHAIN_ID),
      accounts: [HAUPTBUCH_PRIVATE_KEY],
      gas: 8000000,
      gasPrice: 20000000000,
      timeout: 60000
    }
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD"
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
  mocha: {
    timeout: 40000
  }
};
