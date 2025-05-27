const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface TranscriptionResponse {
  message: string;
  file_processed: string;
  transcription: string;
}

export interface GPUInfo {
  name: string;
  memory_total_gb: number | string;
  cuda_capability?: [number, number] | null;
  notes?: string | null;
}

export interface SystemHardwareResponse {
  os_platform?: string | null;
  os_version?: string | null;
  os_release?: string | null;
  architecture?: string | null;
  cpu_model?: string | null;
  cpu_physical_cores?: number | null;
  cpu_logical_cores?: number | null;
  cpu_frequency_max_ghz?: number | string | null;
  ram_total_gb?: number | null;
  ram_available_gb?: number | null;
  cuda_available?: boolean | null;
  cuda_version?: string | null;
  gpu_count?: number | null;
  gpus: GPUInfo[];
  error?: string | null;
  error_partial_results?: string | null;
}

export interface ModelInfoResponse {
  model_name: string;
  status: string;
  input_requirements?: string | null;
  output_type?: string | null;
  license?: string | null;
  note?: string | null;
}

export interface ApiErrorResponse {
  detail: string | { msg: string; type: string }[];
}

/**
 * Transcribes an audio file using the API.
 * @param formData FormData containing the file and transcription options.
 * @returns The transcription response.
 */
export const transcribeAudio = async (
  formData: FormData
): Promise<TranscriptionResponse> => {
  const response = await fetch(`${BASE_URL}/transcribe/`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorData: ApiErrorResponse = await response.json();
    let errorMessage = "Failed to transcribe audio.";
    if (typeof errorData.detail === 'string') {
      errorMessage = errorData.detail;
    } else if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
      errorMessage = errorData.detail.map(err => `${err.msg} (for ${err.type})`).join(', ');
    }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<TranscriptionResponse>;
};

/**
 * Fetches system hardware specifications from the API.
 * @returns The system hardware specifications.
 */
export const getHardwareInfo = async (): Promise<SystemHardwareResponse> => {
  const response = await fetch(`${BASE_URL}/info/system-hardware/`);
  if (!response.ok) {
    const errorData: ApiErrorResponse = await response.json();
    throw new Error(
      typeof errorData.detail === "string"
        ? errorData.detail
        : "Failed to fetch hardware info."
    );
  }
  return response.json() as Promise<SystemHardwareResponse>;
};

/**
 * Fetches ASR model information from the API.
 * @returns The ASR model information.
 */
export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const response = await fetch(`${BASE_URL}/info/asr-model/`);
  if (!response.ok) {
    const errorData: ApiErrorResponse = await response.json();
    throw new Error(
      typeof errorData.detail === "string"
        ? errorData.detail
        : "Failed to fetch model info."
    );
  }
  return response.json() as Promise<ModelInfoResponse>;
}; 