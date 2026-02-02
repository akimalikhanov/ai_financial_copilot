import React, { useState, useEffect } from 'react';
import { UploadCloud, File, Check, X, Loader2 } from 'lucide-react';
import { Button, Input, Card } from './ui';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (meta: any) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUpload }) => {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isOpen) {
        setStep(1);
        setFile(null);
        setProgress(0);
    }
  }, [isOpen]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
    }
  };

  const simulateUpload = () => {
    setStep(3);
    let p = 0;
    const interval = setInterval(() => {
      p += 10;
      setProgress(p);
      if (p >= 100) {
        clearInterval(interval);
        setTimeout(() => {
            onUpload({ title: file?.name, company: 'Uploaded Corp', year: 2024, type: 'Report' });
            onClose();
        }, 500);
      }
    }, 200);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/80 backdrop-blur-sm p-4">
      <Card className="w-full max-w-lg overflow-hidden bg-zinc-900 border-zinc-800 shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <h3 className="text-sm font-bold font-mono tracking-widest text-zinc-100 uppercase">
            Upload Document <span className="text-zinc-600 ml-2">step {step}/3</span>
          </h3>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X size={18} />
          </Button>
        </div>

        <div className="p-6">
          {step === 1 && (
            <div 
              className="border-2 border-dashed border-zinc-700 rounded-lg p-10 flex flex-col items-center justify-center text-center hover:bg-zinc-800/50 transition-colors cursor-pointer"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              <input 
                id="file-upload" 
                type="file" 
                className="hidden" 
                accept=".pdf" 
                onChange={handleFileChange}
              />
              {file ? (
                <div className="flex flex-col items-center">
                   <File size={48} className="text-accent-500 mb-4" />
                   <p className="font-mono text-zinc-200">{file.name}</p>
                   <p className="text-xs text-zinc-500 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <>
                   <UploadCloud size={48} className="text-zinc-600 mb-4" />
                   <p className="text-zinc-300 font-medium">Drag PDF here or click to browse</p>
                   <p className="text-xs text-zinc-500 mt-2">Max file size 50MB</p>
                </>
              )}
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
                <div>
                    <label className="block text-xs font-mono text-zinc-500 mb-1">COMPANY</label>
                    <Input placeholder="e.g. Acme Corp" defaultValue="Uploaded Corp" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-mono text-zinc-500 mb-1">YEAR</label>
                        <Input type="number" defaultValue="2024" />
                    </div>
                    <div>
                        <label className="block text-xs font-mono text-zinc-500 mb-1">TYPE</label>
                        <select className="flex h-10 w-full rounded-sm border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-100">
                            <option>Annual Report</option>
                            <option>10-K</option>
                            <option>Earnings Call</option>
                        </select>
                    </div>
                </div>
            </div>
          )}

          {step === 3 && (
            <div className="py-8">
                 <div className="flex justify-between text-xs font-mono text-zinc-400 mb-2">
                    <span>UPLOADING & PROCESSING</span>
                    <span>{progress}%</span>
                 </div>
                 <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                        className="h-full bg-accent-500 transition-all duration-300 ease-out"
                        style={{ width: `${progress}%` }}
                    />
                 </div>
                 {progress === 100 && (
                    <div className="flex items-center justify-center mt-6 text-emerald-400 gap-2">
                        <Check size={16} />
                        <span className="font-mono text-sm">COMPLETE</span>
                    </div>
                 )}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-zinc-800 bg-zinc-900/50 flex justify-end gap-2">
           {step === 1 && (
             <Button disabled={!file} onClick={() => setStep(2)}>Next: Metadata</Button>
           )}
           {step === 2 && (
             <Button onClick={simulateUpload}>Confirm Upload</Button>
           )}
        </div>
      </Card>
    </div>
  );
};
