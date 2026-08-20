import axios from 'axios';

// The port must match the Python backend (we set to 8080 previously)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080';

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  preprocessor_loaded: boolean;
}

export interface ModelInfoResponse {
  model_id: string;
  model_type: string;
  feature_count: number;
  classes: Record<string, string>;
  frozen: boolean;
}

export interface PredictionProbabilities {
  LOW: number;
  MEDIUM: number;
  HIGH: number;
}

export interface PredictionResult {
  predicted_class: number;
  predicted_label: string;
  probabilities: PredictionProbabilities;
}

export interface WorkflowMetadata {
  name: string | null;
  job_count: number;
  step_count: number;
}

export interface ExplanationSignal {
  name: string;
  detail: string;
}

export interface ExplanationResult {
  title: string;
  signals: ExplanationSignal[];
}

export interface PredictionResponse {
  prediction: PredictionResult;
  workflow: WorkflowMetadata;
  explanation?: ExplanationResult;
  inference_duration_ms: number;
}

export interface ErrorDetail {
  code: string;
  message: string;
}

export interface ErrorResponse {
  error: ErrorDetail;
}

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const response = await api.get<ModelInfoResponse>('/model-info');
  return response.data;
};

export const predictWorkflow = async (workflowYaml: string): Promise<PredictionResponse> => {
  const response = await api.post<PredictionResponse>('/predict', {
    workflow_yaml: workflowYaml,
  });
  return response.data;
};
