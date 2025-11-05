import React, { useRef, useState } from 'react';
import { uploadECGFile } from '../services/api';  // ✅ Import z api.ts
import ResultDisplay from './ResultDisplay';

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
    threshold: number;  // ✅ Nie optional!
}

interface UploadECGProps {
    onResult?: (result: UploadResponse, signal: number[]) => void;
}

export default function UploadECG({ onResult }: UploadECGProps) {
    const [isDragActive, setIsDragActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<UploadResponse | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = async (file: File) => {
        setError(null);
        setIsLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response: UploadResponse = await uploadECGFile(formData);
            setResult(response);
            setError(null);

            if (onResult) {
                const signal = response.signal_for_plot || [];
                onResult(response, signal);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process file');
            setResult(null);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(true);
    };

    const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        const files = e.dataTransfer?.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    };

    return (
        <div className="flex flex-col gap-6">
            {!result ? (
                <div
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
                        isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
                    }`}
                >
                    <div className="mb-4">
                        <svg className="w-12 h-12 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Drag and drop ECG file or click to select</h3>
                    <p className="text-gray-500 text-sm mb-4">Supported: CSV TXT DAT</p>

                    <input
                        ref={inputRef}
                        type="file"
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleFile(file);
                        }}
                        className="hidden"
                    />

                    <button
                        onClick={() => inputRef.current?.click()}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition disabled:opacity-50"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Processing...' : 'Select File'}
                    </button>
                </div>
            ) : null}

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                    <svg className="w-5 h-5 text-red-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <div>
                        <p className="font-semibold text-red-800">Error</p>
                        <p className="text-red-700 text-sm">{error}</p>
                    </div>
                </div>
            )}

            {result && (
                <ResultDisplay
                    result={{
                        ...result,
                        true_label: '',
                        true_label_id: -1,
                        index: -1,
                        is_correct: false
                    }}
                    signal={result.signal_for_plot}
                />
            )}
        </div>
    );
}
