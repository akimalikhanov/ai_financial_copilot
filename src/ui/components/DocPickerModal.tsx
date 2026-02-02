import React, { useState, useMemo } from 'react';
import { X, Search, Check } from 'lucide-react';
import { Document } from '../types';
import { Button, Input, Card, Badge } from './ui';

interface DocPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  docs: Document[];
  selectedIds: string[];
  onConfirm: (ids: string[]) => void;
}

export const DocPickerModal: React.FC<DocPickerModalProps> = ({ isOpen, onClose, docs, selectedIds, onConfirm }) => {
  const [tempSelected, setTempSelected] = useState<string[]>(selectedIds);
  const [search, setSearch] = useState('');

  React.useEffect(() => {
    if (isOpen) setTempSelected(selectedIds);
  }, [isOpen, selectedIds]);

  const filteredDocs = useMemo(() => {
    let d = docs;
    if (search) {
      d = docs.filter(doc => doc.title.toLowerCase().includes(search.toLowerCase()) || doc.company.toLowerCase().includes(search.toLowerCase()));
    }
    // Show 5 most recent by default if no search (mocking "recent" by taking first 5)
    if (!search) {
        return d.slice(0, 5);
    }
    return d;
  }, [docs, search]);

  const toggleId = (id: string) => {
    setTempSelected(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/80 backdrop-blur-sm p-4">
      <Card className="w-full max-w-lg flex flex-col max-h-[80vh] bg-zinc-900 border-zinc-800 shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <h3 className="text-sm font-bold font-mono tracking-widest text-zinc-100 uppercase">
            Select Documents
          </h3>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X size={18} />
          </Button>
        </div>
        
        <div className="p-4 border-b border-zinc-800">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 text-zinc-500" size={14} />
            <Input 
              placeholder="Search documents..." 
              className="pl-9" 
              value={search}
              onChange={e => setSearch(e.target.value)}
              autoFocus
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
           {filteredDocs.length === 0 && (
             <div className="text-center py-8 text-zinc-500 text-sm">No documents found.</div>
           )}
           {filteredDocs.map(doc => {
             const isSelected = tempSelected.includes(doc.id);
             return (
               <div 
                 key={doc.id}
                 onClick={() => toggleId(doc.id)}
                 className={`flex items-start gap-3 p-3 rounded-md cursor-pointer transition-colors border mb-1 ${isSelected ? 'bg-accent-900/20 border-accent-500/50' : 'border-transparent hover:bg-zinc-800'}`}
               >
                 <div className={`w-5 h-5 rounded-sm border flex items-center justify-center mt-0.5 ${isSelected ? 'bg-accent-500 border-accent-500 text-white' : 'border-zinc-600 bg-zinc-900'}`}>
                    {isSelected && <Check size={12} />}
                 </div>
                 <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-zinc-200 truncate">{doc.title}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-[10px] h-5">{doc.company}</Badge>
                      <span className="text-[10px] text-zinc-500 font-mono">{doc.year}</span>
                    </div>
                 </div>
               </div>
             );
           })}
        </div>

        <div className="p-4 border-t border-zinc-800 bg-zinc-900/50 flex justify-end gap-2">
           <Button variant="ghost" onClick={onClose}>Cancel</Button>
           <Button onClick={() => onConfirm(tempSelected)}>
             Add Selected ({tempSelected.length})
           </Button>
        </div>
      </Card>
    </div>
  );
};