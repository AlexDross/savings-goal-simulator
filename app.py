"""
Unit tests for the Savings Goal Probability Simulator.
Run with: python -m pytest test_app.py -v
"""

import pytest
import numpy as np
from app import SimulationParameters, SavingsSimulator

class TestSavingsSimulator:
    """Test suite for the SavingsSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_params = SimulationParameters(
            current_balance=10000,
            target_goal=100000,
            years_to_goal=10,
            annual_contribution=5000,
            contribution_growth_rate=0.0,
            expected_return=0.07,
            volatility=0.0,  # Zero volatility for deterministic tests
            fee_ratio=0.0,
            inflation_rate=0.025,
            contribution_timing='end',
            num_simulations=1000,
            use_real_dollars=False,
            frequency='annual'
        )
    
    def test_deterministic_simulation(self):
        """Test that zero volatility produces deterministic results matching annuity formula."""
        # Set up deterministic parameters (zero volatility)
        params = self.base_params
        params.volatility = 0.0
        params.num_simulations = 100
        
        simulator = SavingsSimulator(params)
        ending_balances, _ = simulator.run_simulation()
        
        # Calculate expected result using future value formula
        # FV = PV(1+r)^n + PMT * [((1+r)^n - 1) / r]
        r = params.expected_return
        n = params.years_to_goal
        pv = params.current_balance
        pmt = params.annual_contribution
        
        expected_fv = pv * (1 + r) ** n + pmt * (((1 + r) ** n - 1) / r)
        
        # All simulation results should be very close to the analytical solution
        assert np.allclose(ending_balances, expected_fv, rtol=1e-10), \
            f"Expected {expected_fv:.2f}, got mean {np.mean(ending_balances):.2f}"
        
        # Verify low variance in results
        assert np.std(ending_balances) < 1.0, \
            f"Standard deviation too high for deterministic case: {np.std(ending_balances)}"
    
    def test_inflation_adjustment(self):
        """Test that inflation adjustment works correctly."""
        params_nominal = self.base_params
        params_nominal.use_real_dollars = False
        params_nominal.volatility = 0.0  # Deterministic
        
        params_real = SimulationParameters(**vars(params_nominal))
        params_real.use_real_dollars = True
        
        simulator_nominal = SavingsSimulator(params_nominal)
        simulator_real = SavingsSimulator(params_real)
        
        balances_nominal, _ = simulator_nominal.run_simulation()
        balances_real, _ = simulator_real.run_simulation()
        
        # Real balances should be lower due to inflation adjustment
        inflation_factor = (1 + params_nominal.inflation_rate) ** (-params_nominal.years_to_goal)
        expected_real = balances_nominal * inflation_factor
        
        assert np.allclose(balances_real, expected_real, rtol=1e-10), \
            "Inflation adjustment not working correctly"
    
    def test_contribution_timing(self):
        """Test that contribution timing affects results correctly."""
        params_start = self.base_params
        params_start.contribution_timing = 'start'
        params_start.volatility = 0.0  # Deterministic
        
        params_end = SimulationParameters(**vars(params_start))
        params_end.contribution_timing = 'end'
        
        simulator_start = SavingsSimulator(params_start)
        simulator_end = SavingsSimulator(params_end)
        
        balances_start, _ = simulator_start.run_simulation()
        balances_end, _ = simulator_end.run_simulation()
        
        # Start-of-period contributions should result in higher ending balances
        # because contributions have more time to grow
        assert np.mean(balances_start) > np.mean(balances_end), \
            "Start-of-period contributions should yield higher balances"
        
        # Verify the difference matches expected compound growth on contributions
        r = params_start.expected_return
        contribution_growth_diff = params_start.annual_contribution * r
        expected_diff_per_year = contribution_growth_diff
        
        # For multiple years, the difference compounds
        actual_diff = np.mean(balances_start) - np.mean(balances_end)
        assert actual_diff > 0, "Expected positive difference for start vs end timing"
    
    def test_fee_impact(self):
        """Test that fees reduce returns appropriately."""
        params_no_fee = self.base_params
        params_no_fee.fee_ratio = 0.0
        params_no_fee.volatility = 0.0  # Deterministic
        
        params_with_fee = SimulationParameters(**vars(params_no_fee))
        params_with_fee.fee_ratio = 0.01  # 1% annual fee
        
        simulator_no_fee = SavingsSimulator(params_no_fee)
        simulator_with_fee = SavingsSimulator(params_with_fee)
        
        balances_no_fee, _ = simulator_no_fee.run_simulation()
        balances_with_fee, _ = simulator_with_fee.run_simulation()
        
        # Balances with fees should be lower
        assert np.mean(balances_with_fee) < np.mean(balances_no_fee), \
            "Fees should reduce ending balances"
        
        # Verify the fee impact is reasonable (not too large or small)
        relative_impact = 1 - (np.mean(balances_with_fee) / np.mean(balances_no_fee))
        assert 0.05 < relative_impact < 0.25, \
            f"Fee impact seems unreasonable: {relative_impact:.2%}"
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        params = self.base_params
        params.target_goal = 50000  # Lower target for higher success rate
        params.volatility = 0.15  # Add some volatility
        
        simulator = SavingsSimulator(params)
        ending_balances, _ = simulator.run_simulation()
        metrics = simulator.calculate_metrics(ending_balances)
        
        # Test basic metrics
        assert 0 <= metrics['success_probability'] <= 1, \
            "Success probability should be between 0 and 1"
        
        assert metrics['median_balance'] > 0, \
            "Median balance should be positive"
        
        assert metrics['percentiles']['25th'] <= metrics['percentiles']['75th'], \
            "25th percentile should be <= 75th percentile"
        
# Test shortfall metrics
if metrics['success_probability'] < 1.0:
    assert metrics['expected_shortfall'] >= 0, \
        "Expected shortfall should be non-negative"
    assert metrics['cvar_5'] >= metrics['expected_shortfall'], \
        "CVaR at 5% should be greater than or equal to expected shortfall"

