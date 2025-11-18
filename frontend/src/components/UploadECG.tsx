import React, { useRef, useState } from 'react';
import { uploadECGFile } from '../services/api';
import ResultDisplay from './ResultDisplay';
import { TestSample } from '../types';

interface UploadResponse {
    predicted_class: string;
    confidence: number;
    is_uncertain: boolean;
    Normal: number;
    Supraventricular: number;
    Ventricular: number;
    Fusion: number;
    Unknown: number;
    signal_for_plot: number[];
    signal_normalized: number[];
    threshold: number;
}

interface UploadECGProps {
    onResult?: (result: TestSample, signal: number[]) => void;
}

export default function UploadECG({ onResult }: UploadECGProps) {
    const [isDragActive, setIsDragActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<TestSample | null>(null);
    const [signal, setSignal] = useState<number[]>([]);
    const inputRef = useRef<HTMLInputElement | null>(null);

    const handleFile = async (file: File) => {
        setError(null);
        setIsLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response: UploadResponse = await uploadECGFile(formData);

            const convertedResult: TestSample = {
                predicted_class: response.predicted_class,
                confidence: response.confidence,
                is_uncertain: response.is_uncertain,
                Normal: response.Normal,
                Supraventricular: response.Supraventricular,
                Ventricular: response.Ventricular,
                Fusion: response.Fusion,
                Unknown: response.Unknown,
                threshold: response.threshold,
                true_label: '',
                true_label_id: 0,
                index: 0,
                is_correct: false,
            };

            setResult(convertedResult);
            setSignal(response.signal_for_plot);
            setError(null);

            if (onResult) {
                onResult(convertedResult, response.signal_for_plot);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process file');
            setResult(null);
            setSignal([]);
        } finally {
            setIsLoading(false);
        }
    };

    // Reszta tak jak poprzednio: obsługa drag&drop itp.

    return (
        <div
            onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragActive(true); }}
            onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragActive(false); }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragActive(false);
                const files = e.dataTransfer?.files;
                if (files && files[0]) handleFile(files[0]);
            }}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
                isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300'
            }`}
            onClick={() => inputRef.current?.click()}
        >
            {!result && (
                <>
                    <p className="mb-2 font-semibold">Drag and drop ECG file or click to select</p>
                    <p className="mb-4 text-sm text-gray-500">Supported: CSV TXT DAT</p>
                    <input
                        type="file"
                        accept=".csv,.txt,.dat"
                        ref={inputRef}
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleFile(file);
                        }}
                        className="hidden"
                    />
                    <button
                        type="button"
                        className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition disabled:opacity-50"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Processing...' : 'Select File'}
                    </button>
                </>
            )}

            {error && (
                <p className="mt-4 text-red-600 font-semibold">Error: {error}</p>
            )}

            {result && (
                <ResultDisplay result={result} signal={signal} />
            )}
        </div>
    );
}
