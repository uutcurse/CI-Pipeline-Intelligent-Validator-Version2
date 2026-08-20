import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import App from './App';
import * as api from './services/api';

vi.mock('./services/api', () => ({
  checkHealth: vi.fn(),
  getModelInfo: vi.fn(),
  predictWorkflow: vi.fn(),
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (api.checkHealth as any).mockResolvedValue({ status: 'UP', model_loaded: true, preprocessor_loaded: true });
    (api.getModelInfo as any).mockResolvedValue({
      model_id: 'E06',
      model_type: 'TF-IDF + Structural Logistic Regression',
      feature_count: 80,
      classes: { '0': 'LOW', '1': 'MEDIUM', '2': 'HIGH' },
      frozen: true,
    });
  });

  it('renders correctly initially', async () => {
    render(<App />);
    expect(screen.getByText('CI Workflow Risk Validator')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByTestId('backend-status')).toHaveTextContent(/Connected/);
    });
  });

  it('handles backend offline', async () => {
    (api.checkHealth as any).mockRejectedValue(new Error('Network error'));
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByTestId('backend-status')).toHaveTextContent(/Offline/);
    });
  });

  it('handles analyze success', async () => {
    (api.predictWorkflow as any).mockResolvedValue({
      prediction: {
        predicted_class: 2,
        predicted_label: 'HIGH',
        probabilities: { LOW: 0.1, MEDIUM: 0.15, HIGH: 0.75 }
      },
      workflow: { name: 'CI', job_count: 1, step_count: 5 },
      inference_duration_ms: 10
    });

    render(<App />);
    const btn = screen.getByTestId('analyze-button');
    fireEvent.click(btn);
    
    await waitFor(() => {
      expect(screen.getByTestId('prediction-result')).toBeInTheDocument();
      expect(screen.getAllByText('HIGH').length).toBeGreaterThan(0);
    });
  });

  it('handles API error', async () => {
    (api.predictWorkflow as any).mockRejectedValue({
      response: { data: { error: { message: 'Invalid workflow syntax' } } }
    });

    render(<App />);
    const btn = screen.getByTestId('analyze-button');
    fireEvent.click(btn);
    
    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
      expect(screen.getByText(/Invalid workflow syntax/)).toBeInTheDocument();
    });
  });
});
