import React from 'react';

interface AboutModalProps {
    isOpen: boolean;
    onClose: () => void;
}

// SVG Icons
const CloseIcon = () => (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
);

const ExternalLinkIcon = () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
    </svg>
);

const GitHubIcon = () => (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
);

const GitHubIconLarge = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
);

const AboutModal: React.FC<AboutModalProps> = ({ isOpen, onClose }) => {
    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={onClose}
        >
            <div
                className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto scrollbar-hide"
                onClick={(e) => e.stopPropagation()}
                style={{
                    scrollbarWidth: 'none',
                    msOverflowStyle: 'none',
                }}
            >
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-6 rounded-t-xl">
                    <div className="flex items-center justify-between">
                        <h2 className="text-2xl font-bold text-white">About the Project</h2>
                        <button
                            onClick={onClose}
                            className="text-white hover:text-gray-200 transition-colors"
                        >
                            <CloseIcon />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="p-8 space-y-6">
                    {/* Project Description */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">ECG Arrhythmia Classifier</h3>
                        <p className="text-gray-700 leading-relaxed">
                            A project developed for the Computer Science in Medicine Industry course, focused on
                            machine learning for automatic detection and classification of cardiac arrhythmias from
                            ECG signals. The system uses deep learning models trained on the MIT-BIH Arrhythmia
                            Database to identify five different types of heartbeat patterns: Normal, Supraventricular,
                            Ventricular, Fusion, and Unknown beats.
                        </p>
                    </section>

                    {/* Dataset */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">Dataset</h3>
                        <div className="bg-gray-50 rounded-lg p-4">
                            <p className="text-gray-700 leading-relaxed mb-3">
                                <strong>MIT-BIH Arrhythmia Database</strong> - A widely used benchmark dataset
                                containing 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47
                                subjects. The database includes annotations for over 110,000 heartbeats, classified
                                by expert cardiologists.
                            </p>
                            <a
                                href="https://www.kaggle.com/datasets/shayanfazeli/heartbeat"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center gap-1"
                            >
                                <span className="font-semibold">Source:</span> View on Kaggle
                                <ExternalLinkIcon />
                            </a>
                        </div>
                    </section>

                    {/* Model Architecture */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">Model Architecture</h3>
                        <p className="text-gray-700 leading-relaxed mb-4">
                            The system employs a hybrid <strong>CNN-LSTM architecture</strong> that combines
                            Convolutional Neural Networks for feature extraction with Long Short-Term Memory
                            networks for temporal pattern recognition. Signals are preprocessed using StandardScaler
                            normalization before classification.
                        </p>

                        {/* Quality Metrics */}
                        <div className="bg-gray-50 rounded-lg p-4">
                            <h4 className="font-semibold text-gray-800 mb-3">Quality Metrics - Normal vs Abnormal</h4>
                            <img
                                src="/qm.png"
                                alt="Model Quality Metrics"
                                className="w-full rounded-lg shadow-md"
                            />
                        </div>
                    </section>

                    {/* Source Code */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">Source Code</h3>
                        <a
                            href="https://github.com/budziun/ecg-project"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 px-6 py-3 bg-gray-900 text-white font-semibold rounded-lg hover:bg-gray-800 transition-colors"
                        >
                            <GitHubIconLarge />
                            View on GitHub
                        </a>
                    </section>

                    {/* Team */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">Team</h3>
                        <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6">
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <img
                                            src="https://github.com/MacSwider.png"
                                            alt="Maciej Świder"
                                            className="w-10 h-10 rounded-full object-cover"
                                            onError={(e) => {
                                                e.currentTarget.style.display = 'none';
                                                e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                            }}
                                        />
                                        <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm hidden">
                                            MS
                                        </div>
                                        <div>
                                            <p className="font-semibold text-gray-800">Maciej Świder</p>
                                            <p className="text-sm text-gray-600">Project Manager, Data Scientist</p>
                                        </div>
                                    </div>
                                    <a
                                        href="https://github.com/MacSwider"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-4 py-2 bg-gray-900 text-white text-sm rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-2"
                                    >
                                        <GitHubIcon />
                                        GitHub
                                    </a>
                                </div>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <img
                                            src="https://github.com/budziun.png"
                                            alt="Jakub Budzich"
                                            className="w-10 h-10 rounded-full object-cover"
                                            onError={(e) => {
                                                e.currentTarget.style.display = 'none';
                                                e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                            }}
                                        />
                                        <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm hidden">
                                            JB
                                        </div>
                                        <div>
                                            <p className="font-semibold text-gray-800">Jakub Budzich</p>
                                            <p className="text-sm text-gray-600">ML/Web Engineer, Tech Lead</p>
                                        </div>
                                    </div>
                                    <a
                                        href="https://github.com/budziun"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-4 py-2 bg-gray-900 text-white text-sm rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-2"
                                    >
                                        <GitHubIcon />
                                        GitHub
                                    </a>
                                </div>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <img
                                            src="https://github.com/AdamCzp.png"
                                            alt="Adam Czaplicki"
                                            className="w-10 h-10 rounded-full object-cover"
                                            onError={(e) => {
                                                e.currentTarget.style.display = 'none';
                                                e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                            }}
                                        />
                                        <div className="w-10 h-10 bg-indigo-500 rounded-full flex items-center justify-center text-white font-bold text-sm hidden">
                                            AC
                                        </div>
                                        <div>
                                            <p className="font-semibold text-gray-800">Adam Czaplicki</p>
                                            <p className="text-sm text-gray-600">UX Designer, QA Specialist</p>
                                        </div>
                                    </div>
                                    <a
                                        href="https://github.com/AdamCzp"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-4 py-2 bg-gray-900 text-white text-sm rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-2"
                                    >
                                        <GitHubIcon />
                                        GitHub
                                    </a>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Technology Stack */}
                    <section>
                        <h3 className="text-xl font-bold text-gray-800 mb-3">Technology Stack</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-purple-50 rounded-lg p-4">
                                <h4 className="font-semibold text-purple-800 mb-2">Machine Learning</h4>
                                <ul className="text-sm text-gray-700 space-y-1">
                                    <li>• PyTorch</li>
                                    <li>• NumPy</li>
                                    <li>• Pandas</li>
                                    <li>• scikit-learn</li>
                                </ul>
                            </div>
                            <div className="bg-green-50 rounded-lg p-4">
                                <h4 className="font-semibold text-green-800 mb-2">Backend</h4>
                                <ul className="text-sm text-gray-700 space-y-1">
                                    <li>• FastAPI (Python)</li>
                                    <li>• Swagger</li>
                                    <li>• Docker</li>
                                </ul>
                            </div>
                            <div className="bg-blue-50 rounded-lg p-4">
                                <h4 className="font-semibold text-blue-800 mb-2">Frontend</h4>
                                <ul className="text-sm text-gray-700 space-y-1">
                                    <li>• React</li>
                                    <li>• Recharts</li>
                                    <li>• Tailwind CSS</li>
                                </ul>
                            </div>
                        </div>
                    </section>
                </div>

                {/* Footer */}
                <div className="bg-gray-50 p-6 rounded-b-xl border-t border-gray-200">
                    <p className="text-sm text-gray-600 text-center">
                        University of Warmia and Mazury in Olsztyn • Computer Science • 2025
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AboutModal;
