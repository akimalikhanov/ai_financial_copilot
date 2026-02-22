import React, { useState, useEffect } from 'react';
import { UploadCloud, File, Check, X, Loader2 } from 'lucide-react';
import { Button, Input, Card } from './ui';
import { uploadDocument, ApiError, type UploadDocumentResponse } from '../services/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (doc: UploadDocumentResponse) => void;
}

const mapStatus = (s: string): 'Ready' | 'Processing' | 'Error' =>
  s === 'ready' ? 'Ready' : s === 'failed' ? 'Error' : 'Processing';

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUpload }) => {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [company, setCompany] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setStep(1);
      setFile(null);
      setCompany('');
      setUploading(false);
      setError(null);
    }
  }, [isOpen]);

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleConfirmUpload = async () => {
    if (!file) return;
    setStep(3);
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (company.trim()) formData.append('company', company.trim());
      const doc = await uploadDocument(formData);
      onUpload({ ...doc, status: mapStatus(doc.status) });
      onClose();
    } catch (err) {
      setError((err as ApiError)?.message ?? 'Upload failed');
      setStep(2);
    } finally {
      setUploading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <Card className="w-full max-w-lg overflow-hidden shadow-xl" variant="elevated">
        <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-semibold text-[var(--text)] uppercase tracking-wide">
            Upload Document <span className="text-[var(--text-faint)] ml-2 font-normal">step {step}/3</span>
          </h3>
          <Button variant="ghost" size="icon" onClick={onClose} disabled={uploading}>
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
                <Input
                  placeholder="e.g. Acme Corp"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                />
              </div>
              {error && (
                <p className="text-sm text-[var(--danger)]">{error}</p>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="py-8 flex flex-col items-center gap-4">
              {uploading ? (
                <>
                  <Loader2 size={32} className="animate-spin text-[var(--accent)]" />
                  <p className="text-sm text-[var(--text-muted)]">Uploading & processing…</p>
                </>
              ) : (
                <>
                  <Check size={32} className="text-[var(--success)]" />
                  <p className="text-sm font-medium text-[var(--text)]">Complete</p>
                </>
              )}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-1)] flex justify-end gap-2">
          {step === 1 && (
            <Button disabled={!file} onClick={() => setStep(2)}>Next: Metadata</Button>
          )}
          {step === 2 && (
            <Button onClick={handleConfirmUpload} disabled={uploading}>Confirm Upload</Button>
          )}
        </div>
      </Card>
    </div>
  );
};
