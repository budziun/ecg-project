import { ECGPrediction, TestSamplesResponse, BatchPredictionResponse } from '../types';

const getApiUrl = (): string => {
    const host = window.location.hostname;
    const port = 8000;
    return `http://${host}:${port}`;
};

const API_BASE = getApiUrl();

export const api = {
    getTestSamples: async (count: number = 5): Promise<TestSamplesResponse> => {
        const response = await fetch(`${API_BASE}/test-samples?count=${count}`);
        if (!response.ok) throw new Error('Failed to fetch test samples');
        return response.json();
    },

    predictCSV: async (file: File): Promise<BatchPredictionResponse> => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/batch-predict-csv`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Failed to predict');
        return response.json();
    },

    predict: async (signal: number[]): Promise<ECGPrediction> => {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ signal })
        });
        if (!response.ok) throw new Error('Failed to predict');
        return response.json();
    },

    uploadECGFile: async (formData: FormData) => {
        const response = await fetch(`${API_BASE}/upload-csv`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Failed to upload ECG file');

        const data = await response.json();

        return {
            predicted_class: data.predicted_class || 'Unknown',
            confidence: data.confidence || 0,
            is_uncertain: data.is_uncertain ?? false,
            Normal: data.Normal ?? 0,
            Supraventricular: data.Supraventricular ?? 0,
            Ventricular: data.Ventricular ?? 0,
            Fusion: data.Fusion ?? 0,
            Unknown: data.Unknown ?? 0,
            signal_raw: data.signal_raw || [],              // ✅ ZMIANA
            signal_normalized: data.signal_normalized || [],
            threshold: data.threshold ?? 0
        };
    }
};

export const uploadECGFile = api.uploadECGFile;
