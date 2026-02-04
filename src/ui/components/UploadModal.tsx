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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <Card className="w-full max-w-lg overflow-hidden shadow-xl" variant="elevated">
        <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-semibold text-[var(--text)] uppercase tracking-wide">
            Upload Document <span className="text-[var(--text-faint)] ml-2 font-normal">step {step}/3</span>
          </h3>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X size={18} />
          </Button>
        </div>

        <div className="p-6">
          {step === 1 && (
            <div
              className="border-2 border-dashed border-[var(--border)] rounded-xl p-10 flex flex-col items-center justify-center text-center hover:bg-[var(--surface-2)] hover:border-[var(--accent)] transition-colors cursor-pointer"
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
                   <File size={48} className="text-[var(--accent)] mb-4" />
                   <p className="font-medium text-[var(--text)]">{file.name}</p>
                   <p className="text-xs text-[var(--text-faint)] mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <>
                   <UploadCloud size={48} className="text-[var(--text-faint)] mb-4" />
                   <p className="text-[var(--text)] font-medium">Drag PDF here or click to browse</p>
                   <p className="text-xs text-[var(--text-faint)] mt-2">Max file size 50MB</p>
                </>
              )}
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
                <div>
                    <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5 uppercase tracking-wide">Company</label>
                    <Input placeholder="e.g. Acme Corp" defaultValue="Uploaded Corp" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5 uppercase tracking-wide">Year</label>
                        <Input type="number" defaultValue="2024" />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5 uppercase tracking-wide">Type</label>
                        <select className="flex h-10 w-full rounded-md border border-[var(--input-border)] bg-[var(--input-bg)] px-3 py-2 text-sm text-[var(--text)] focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)] outline-none cursor-pointer">
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
                 <div className="flex justify-between text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
                    <span>Uploading & Processing</span>
                    <span className="font-mono">{progress}%</span>
                 </div>
                 <div className="h-2 w-full bg-[var(--surface-3)] rounded-full overflow-hidden">
                    <div
                        className="h-full bg-[var(--accent)] transition-all duration-300 ease-out rounded-full"
                        style={{ width: `${progress}%` }}
                    />
                 </div>
                 {progress === 100 && (
                    <div className="flex items-center justify-center mt-6 text-[var(--success)] gap-2">
                        <Check size={16} />
                        <span className="font-medium text-sm">Complete</span>
                    </div>
                 )}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-1)] flex justify-end gap-2">
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
