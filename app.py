"""
Savings Goal Probability Simulator
A production-quality Streamlit application for Monte Carlo simulation of savings goals.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Dict, List
import io
import base64

# Set page config
st.set_page_config(
    page_title="Savings Goal Probability Simulator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation."""
    current_balance: float
    target_goal: float
    years_to_goal: float
    annual_contribution: float
    contribution_growth_rate: float
    expected_return: float
    volatility: float
    fee_ratio: float
    inflation_rate: float
    contribution_timing: str  # 'start' or 'end'
    num_simulations: int
    use_real_dollars: bool
    frequency: str  # 'annual' or 'monthly'

class SavingsSimulator:
    """Monte Carlo simulation engine for savings goals."""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        
    def run_simulation(self) -> np.ndarray:
        """
        Run Monte Carlo simulation and return array of ending balances.
        
        Returns:
            np.ndarray: Array of ending balances for each simulation path
        """
        # Determine periods per year and adjust parameters
        if self.params.frequency == 'monthly':
            periods_per_year = 12
            total_periods = int(self.params.years_to_goal * 12)
            period_return = self.params.expected_return / 12
            period_volatility = self.params.volatility / np.sqrt(12)
            period_fee = self.params.fee_ratio / 12
            period_contribution = self.params.annual_contribution / 12
        else:  # annual
            periods_per_year = 1
            total_periods = int(self.params.years_to_goal)
            period_return = self.params.expected_return
            period_volatility = self.params.volatility
            period_fee = self.params.fee_ratio
            period_contribution = self.params.annual_contribution
        
        # Initialize arrays
        num_sims = self.params.num_simulations
        balances = np.full((num_sims, total_periods + 1), self.params.current_balance)
        
        # Generate random returns (log-normal distribution)
        # Adjust for fees
        adjusted_return = period_return - period_fee - 0.5 * period_volatility**2
        random_returns = np.random.normal(
            adjusted_return, period_volatility, (num_sims, total_periods)
        )
        
        # Convert to actual returns
        returns = np.exp(random_returns)
        
        # Calculate contributions over time (with growth)
        contributions = np.zeros((num_sims, total_periods))
        for t in range(total_periods):
            year = t / periods_per_year
            growth_factor = (1 + self.params.contribution_growth_rate) ** year
            contributions[:, t] = period_contribution * growth_factor
        
        # Run simulation
        for t in range(total_periods):
            if self.params.contribution_timing == 'start':
                # Add contribution at start of period, then apply return
                balances[:, t + 1] = (balances[:, t] + contributions[:, t]) * returns[:, t]
            else:
                # Apply return first, then add contribution at end
                balances[:, t + 1] = balances[:, t] * returns[:, t] + contributions[:, t]
        
        ending_balances = balances[:, -1]
        
        # Adjust for inflation if using real dollars
        if self.params.use_real_dollars:
            inflation_factor = (1 + self.params.inflation_rate) ** (-self.params.years_to_goal)
            ending_balances = ending_balances * inflation_factor
        
        return ending_balances, balances
    
    def calculate_metrics(self, ending_balances: np.ndarray) -> Dict:
        """
        Calculate key metrics from simulation results.
        
        Args:
            ending_balances: Array of ending balances from simulation
            
        Returns:
            Dict: Dictionary of calculated metrics
        """
        # Adjust target goal for inflation if using real dollars
        target = self.params.target_goal
        if self.params.use_real_dollars:
            target = target / ((1 + self.params.inflation_rate) ** self.params.years_to_goal)
        
        success_outcomes = ending_balances >= target
        success_probability = np.mean(success_outcomes)
        
        # Basic statistics
        median_balance = np.median(ending_balances)
        mean_balance = np.mean(ending_balances)
        
        # Percentiles
        percentiles = {
            '10th': np.percentile(ending_balances, 10),
            '25th': np.percentile(ending_balances, 25),
            '75th': np.percentile(ending_balances, 75),
            '90th': np.percentile(ending_balances, 90)
        }
        
        # Shortfall analysis
        shortfall_outcomes = ending_balances[~success_outcomes]
        if len(shortfall_outcomes) > 0:
            expected_shortfall = np.mean(target - shortfall_outcomes)
            # CVaR at 5% level (worst 5% of shortfall cases)
            cvar_5 = np.mean(target - np.percentile(shortfall_outcomes, 5))
        else:
            expected_shortfall = 0
            cvar_5 = 0
        
        return {
            'success_probability': success_probability,
            'median_balance': median_balance,
            'mean_balance': mean_balance,
            'percentiles': percentiles,
            'expected_shortfall': expected_shortfall,
            'cvar_5': cvar_5,
            'target_adjusted': target
        }

def create_fan_chart(balances: np.ndarray, params: SimulationParameters) -> go.Figure:
    """Create a fan chart showing percentile bands over time."""
    periods = balances.shape[1]
    time_axis = np.arange(periods) / (12 if params.frequency == 'monthly' else 1)
    
    # Calculate percentiles over time
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    data = {}
    
    for p in percentiles:
        data[f'{p}th'] = np.percentile(balances, p, axis=0)
    
    fig = go.Figure()
    
    # Add percentile bands
    colors = ['rgba(255,0,0,0.1)', 'rgba(255,100,0,0.2)', 'rgba(255,200,0,0.3)',
              'rgba(0,100,255,0.4)', 'rgba(255,200,0,0.3)', 'rgba(255,100,0,0.2)', 'rgba(255,0,0,0.1)']
    
    # Fill between percentiles
    for i in range(len(percentiles)//2):
        lower_p = percentiles[i]
        upper_p = percentiles[-(i+1)]
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_axis, time_axis[::-1]]),
            y=np.concatenate([data[f'{lower_p}th'], data[f'{upper_p}th'][::-1]]),
            fill='toself',
            fillcolor=colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{lower_p}th-{upper_p}th percentile',
            hoverinfo='skip'
        ))
    
    # Add median line
    fig.add_trace(go.Scatter(
        x=time_axis, y=data['50th'],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Median (50th percentile)'
    ))
    
    # Add target goal line
    target = params.target_goal
    if params.use_real_dollars:
        target = target / ((1 + params.inflation_rate) ** params.years_to_goal)
    
    fig.add_hline(y=target, line_dash="dash", line_color="red",
                  annotation_text=f"Target: ${target:,.0f}")
    
    fig.update_layout(
        title="Balance Projection Fan Chart",
        xaxis_title="Years",
        yaxis_title="Balance ($)",
        showlegend=True,
        height=500
    )
    
    return fig

def create_histogram(ending_balances: np.ndarray, target: float) -> go.Figure:
    """Create histogram of ending balances."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ending_balances,
        nbinsx=50,
        name='Ending Balances',
        opacity=0.7
    ))
    
    fig.add_vline(x=target, line_dash="dash", line_color="red",
                  annotation_text=f"Target: ${target:,.0f}")
    
    fig.add_vline(x=np.median(ending_balances), line_dash="solid", line_color="blue",
                  annotation_text=f"Median: ${np.median(ending_balances):,.0f}")
    
    fig.update_layout(
        title="Distribution of Ending Balances",
        xaxis_title="Ending Balance ($)",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def sensitivity_analysis(base_params: SimulationParameters) -> pd.DataFrame:
    """Perform sensitivity analysis on key parameters."""
    base_simulator = SavingsSimulator(base_params)
    base_results, _ = base_simulator.run_simulation()
    base_metrics = base_simulator.calculate_metrics(base_results)
    base_success = base_metrics['success_probability']
    
    # Parameters to test
    sensitivity_params = {
        'Annual Contribution': ('annual_contribution', 0.1),
        'Expected Return': ('expected_return', 0.01),
        'Volatility': ('volatility', 0.01),
        'Fees': ('fee_ratio', 0.001),
        'Years to Goal': ('years_to_goal', base_params.years_to_goal * 0.1),
    }
    
    results = []
    
    for param_name, (param_attr, shock_size) in sensitivity_params.items():
        # Test positive shock
        params_up = SimulationParameters(**vars(base_params))
        setattr(params_up, param_attr, getattr(base_params, param_attr) + shock_size)
        simulator_up = SavingsSimulator(params_up)
        results_up, _ = simulator_up.run_simulation()
        metrics_up = simulator_up.calculate_metrics(results_up)
        
        # Test negative shock
        params_down = SimulationParameters(**vars(base_params))
        setattr(params_down, param_attr, getattr(base_params, param_attr) - shock_size)
        simulator_down = SavingsSimulator(params_down)
        results_down, _ = simulator_down.run_simulation()
        metrics_down = simulator_down.calculate_metrics(results_down)
        
        results.append({
            'Parameter': param_name,
            'Base Success %': base_success * 100,
            'Success % (+10%)': metrics_up['success_probability'] * 100,
            'Success % (-10%)': metrics_down['success_probability'] * 100,
            'Impact (+10%)': (metrics_up['success_probability'] - base_success) * 100,
            'Impact (-10%)': (metrics_down['success_probability'] - base_success) * 100,
        })
    
    return pd.DataFrame(results)

def goal_seek_contribution(target_success_prob: float, params: SimulationParameters) -> float:
    """Find required contribution for target success probability."""
    low, high = 0, params.annual_contribution * 5
    tolerance = 0.01
    
    for _ in range(20):  # Max iterations
        mid = (low + high) / 2
        test_params = SimulationParameters(**vars(params))
        test_params.annual_contribution = mid
        
        simulator = SavingsSimulator(test_params)
        results, _ = simulator.run_simulation()
        metrics = simulator.calculate_metrics(results)
        
        if abs(metrics['success_probability'] - target_success_prob) < tolerance:
            return mid
        elif metrics['success_probability'] < target_success_prob:
            low = mid
        else:
            high = mid
    
    return mid

def goal_seek_years(target_success_prob: float, params: SimulationParameters) -> float:
    """Find required years for target success probability."""
    low, high = 1, params.years_to_goal * 2
    tolerance = 0.01
    
    for _ in range(20):  # Max iterations
        mid = (low + high) / 2
        test_params = SimulationParameters(**vars(params))
        test_params.years_to_goal = mid
        
        simulator = SavingsSimulator(test_params)
        results, _ = simulator.run_simulation()
        metrics = simulator.calculate_metrics(results)
        
        if abs(metrics['success_probability'] - target_success_prob) < tolerance:
            return mid
        elif metrics['success_probability'] < target_success_prob:
            low = mid
        else:
            high = mid
    
    return mid

def main():
    """Main Streamlit application."""
    st.title("💰 Savings Goal Probability Simulator")
    st.markdown("*Educational tool for financial planning - Not financial advice*")
    
    # Sidebar inputs
    st.sidebar.header("📊 Simulation Parameters")
    
    # Basic parameters
    st.sidebar.subheader("🎯 Basic Settings")
    current_balance = st.sidebar.number_input("Current Balance ($)", value=10000, min_value=0)
    target_goal = st.sidebar.number_input("Target Goal ($)", value=100000, min_value=1)
    years_to_goal = st.sidebar.slider("Years to Goal", min_value=1, max_value=50, value=10)
    annual_contribution = st.sidebar.number_input("Annual Contribution ($)", value=6000, min_value=0)
    
    # Advanced parameters
    st.sidebar.subheader("🔧 Advanced Settings")
    contribution_growth = st.sidebar.slider("Contribution Growth Rate (%/year)", 
                                          min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
    expected_return = st.sidebar.slider("Expected Annual Return (%)", 
                                       min_value=0.0, max_value=20.0, value=7.0, step=0.1) / 100
    volatility = st.sidebar.slider("Annual Volatility (%)", 
                                  min_value=0.0, max_value=50.0, value=15.0, step=0.5) / 100
    fee_ratio = st.sidebar.slider("Annual Fee Ratio (%)", 
                                 min_value=0.0, max_value=3.0, value=0.5, step=0.05) / 100
    inflation_rate = st.sidebar.slider("Inflation Rate (%)", 
                                      min_value=0.0, max_value=10.0, value=2.5, step=0.1) / 100
    
    contribution_timing = st.sidebar.selectbox("Contribution Timing", 
                                              ["start", "end"], index=0)
    use_real_dollars = st.sidebar.checkbox("Use Real (Inflation-Adjusted) Dollars", value=False)
    frequency = st.sidebar.selectbox("Simulation Frequency", 
                                    ["annual", "monthly"], index=1)
    num_simulations = st.sidebar.selectbox("Number of Simulations", 
                                          [1000, 5000, 10000, 25000], index=2)
    
    # Create parameters object
    params = SimulationParameters(
        current_balance=current_balance,
        target_goal=target_goal,
        years_to_goal=years_to_goal,
        annual_contribution=annual_contribution,
        contribution_growth_rate=contribution_growth,
        expected_return=expected_return,
        volatility=volatility,
        fee_ratio=fee_ratio,
        inflation_rate=inflation_rate,
        contribution_timing=contribution_timing,
        num_simulations=num_simulations,
        use_real_dollars=use_real_dollars,
        frequency=frequency
    )
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            simulator = SavingsSimulator(params)
            ending_balances, all_balances = simulator.run_simulation()
            metrics = simulator.calculate_metrics(ending_balances)
            
            # Store results in session state
            st.session_state['results'] = {
                'ending_balances': ending_balances,
                'all_balances': all_balances,
                'metrics': metrics,
                'params': params
            }
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        metrics = results['metrics']
        
        # KPI Cards
        st.header("📈 Key Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Probability", f"{metrics['success_probability']:.1%}")
        with col2:
            st.metric("Median Balance", f"${metrics['median_balance']:,.0f}")
        with col3:
            st.metric("Expected Shortfall", f"${metrics['expected_shortfall']:,.0f}")
        with col4:
            st.metric("CVaR (5%)", f"${metrics['cvar_5']:,.0f}")
        
        # Tabs for detailed results
        tab1, tab2, tab3 = st.tabs(["📊 Results & KPIs", "📈 Charts", "🎯 Sensitivity & Goal-Seek"])
        
        with tab1:
            st.subheader("Detailed Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Balance Statistics**")
                st.write(f"Mean Balance: ${metrics['mean_balance']:,.0f}")
                st.write(f"Median Balance: ${metrics['median_balance']:,.0f}")
                st.write(f"10th Percentile: ${metrics['percentiles']['10th']:,.0f}")
                st.write(f"25th Percentile: ${metrics['percentiles']['25th']:,.0f}")
                st.write(f"75th Percentile: ${metrics['percentiles']['75th']:,.0f}")
                st.write(f"90th Percentile: ${metrics['percentiles']['90th']:,.0f}")
            
            with col2:
                st.write("**Risk Metrics**")
                st.write(f"Success Probability: {metrics['success_probability']:.1%}")
                st.write(f"Failure Probability: {1-metrics['success_probability']:.1%}")
                st.write(f"Expected Shortfall: ${metrics['expected_shortfall']:,.0f}")
                st.write(f"CVaR at 5%: ${metrics['cvar_5']:,.0f}")
                target_display = metrics['target_adjusted']
                st.write(f"Adjusted Target: ${target_display:,.0f}")
        
        with tab2:
            st.subheader("Balance Projection Over Time")
            fan_chart = create_fan_chart(results['all_balances'], results['params'])
            st.plotly_chart(fan_chart, use_container_width=True)
            
            st.subheader("Distribution of Ending Balances")
            histogram = create_histogram(results['ending_balances'], metrics['target_adjusted'])
            st.plotly_chart(histogram, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Sensitivity Analysis")
                if st.button("Run Sensitivity Analysis"):
                    with st.spinner("Running sensitivity analysis..."):
                        sensitivity_df = sensitivity_analysis(results['params'])
                        st.dataframe(sensitivity_df, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Goal Seek")
                target_prob = st.slider("Target Success Probability (%)", 
                                       min_value=50, max_value=95, value=80) / 100
                
                if st.button("Find Required Contribution"):
                    with st.spinner("Calculating..."):
                        req_contrib = goal_seek_contribution(target_prob, results['params'])
                        st.success(f"Required annual contribution: ${req_contrib:,.0f}")
                
                if st.button("Find Required Years"):
                    with st.spinner("Calculating..."):
                        req_years = goal_seek_years(target_prob, results['params'])
                        st.success(f"Required years: {req_years:.1f}")
        
        # Download buttons
        st.header("💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = pd.DataFrame({
                'Simulation': range(len(results['ending_balances'])),
                'Ending_Balance': results['ending_balances']
            })
            csv = csv_data.to_csv(index=False)
            st.download_button(
                label="📄 Download Results CSV",
                data=csv,
                file_name="savings_simulation_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary report
            summary = f"""
            Savings Goal Simulation Summary
            ==============================
            
            Parameters:
            - Current Balance: ${params.current_balance:,.0f}
            - Target Goal: ${params.target_goal:,.0f}
            - Years to Goal: {params.years_to_goal}
            - Annual Contribution: ${params.annual_contribution:,.0f}
            - Expected Return: {params.expected_return:.1%}
            - Volatility: {params.volatility:.1%}
            - Fees: {params.fee_ratio:.2%}
            
            Results:
            - Success Probability: {metrics['success_probability']:.1%}
            - Median Balance: ${metrics['median_balance']:,.0f}
            - Expected Shortfall: ${metrics['expected_shortfall']:,.0f}
            """
            
            st.download_button(
                label="📋 Download Summary Report",
                data=summary,
                file_name="savings_simulation_summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
