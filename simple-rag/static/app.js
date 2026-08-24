document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const queryTextarea = document.getElementById('query-textarea');
    const btnSubmitQuery = document.getElementById('btn-submit-query');
    const querySpinner = document.getElementById('query-spinner');
    const answerOutput = document.getElementById('answer-output');
    
    const dataDirectoryInput = document.getElementById('data-directory-input');
    const btnIngest = document.getElementById('btn-ingest');
    const ingestSpinner = document.getElementById('ingest-spinner');
    
    const storeStatus = document.getElementById('store-status');
    const chunkCount = document.getElementById('chunk-count');
    const retrievedContextsList = document.getElementById('retrieved-contexts-list');
    const topKSelect = document.getElementById('top-k-select');

    // Document Manager Elements
    const documentList = document.getElementById('document-list');
    const btnNewDoc = document.getElementById('btn-new-doc');
    const editorModal = document.getElementById('editor-modal');
    const closeEditor = document.getElementById('close-editor');
    const btnCancelEdit = document.getElementById('btn-cancel-edit');
    const btnSaveDoc = document.getElementById('btn-save-doc');
    const saveSpinner = document.getElementById('save-spinner');
    const docFilename = document.getElementById('doc-filename');
    const docContent = document.getElementById('doc-content');
    const modalTitle = document.getElementById('modal-title');

    // Chat History & Suggested Elements
    const chatHistoryList = document.getElementById('chat-history-list');
    const btnClearHistory = document.getElementById('btn-clear-history');
    const btnExportChat = document.getElementById('btn-export-chat');
    const tfidfInspector = document.getElementById('tfidf-inspector-container');
    const tokenMatchesList = document.getElementById('token-matches-list');
    const suggestionPills = document.getElementById('suggestion-pills');

    let isEditing = false;
    let chatHistory = JSON.parse(localStorage.getItem('acme_rag_chat_history') || '[]');

    // Load initial system stats & documents
    async function init() {
        await loadStats();
        await loadDocuments();
        renderChatHistory();
    }

    async function loadStats() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                updateStatsUI(data);
            }
        } catch (error) {
            console.error('Error fetching system status:', error);
            storeStatus.textContent = 'Connection Error';
            storeStatus.style.color = '#ef4444';
        }
    }

    function updateStatsUI(data) {
        if (data.exists) {
            storeStatus.textContent = 'Loaded';
            storeStatus.style.color = '#10b981';
            chunkCount.textContent = data.chunk_count;
        } else {
            storeStatus.textContent = 'Not Found';
            storeStatus.style.color = '#f59e0b';
            chunkCount.textContent = '0';
        }
    }

    // List Documents
    async function loadDocuments() {
        const directory = dataDirectoryInput.value.trim() || './data';
        try {
            const response = await fetch(`/api/documents?directory=${encodeURIComponent(directory)}`);
            if (!response.ok) throw new Error('Failed to fetch documents');
            
            const docs = await response.json();
            renderDocuments(docs);
        } catch (error) {
            console.error('Error loading documents:', error);
            documentList.innerHTML = `<div class="empty-state error">Error loading files: ${error.message}</div>`;
        }
    }

    function renderDocuments(docs) {
        if (!docs || docs.length === 0) {
            documentList.innerHTML = '<div class="empty-state">No documents found. Click New to create one!</div>';
            return;
        }

        documentList.innerHTML = '';
        docs.forEach(doc => {
            const row = document.createElement('div');
            row.className = 'doc-item';
            
            const info = document.createElement('div');
            info.className = 'doc-info';
            
            const name = document.createElement('div');
            name.className = 'doc-name';
            name.textContent = doc.name;
            name.title = doc.rel_path;
            
            const size = document.createElement('div');
            size.className = 'doc-size';
            size.textContent = `${(doc.size / 1024).toFixed(2)} KB`;
            
            info.appendChild(name);
            info.appendChild(size);
            
            const actions = document.createElement('div');
            actions.className = 'doc-actions';
            
            const editBtn = document.createElement('button');
            editBtn.className = 'btn-icon-sm edit';
            editBtn.innerHTML = '✏️ Edit';
            editBtn.onclick = () => openDocEditor(doc.rel_path);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn-icon-sm delete';
            deleteBtn.innerHTML = '🗑️';
            deleteBtn.onclick = () => deleteDoc(doc.rel_path);
            
            actions.appendChild(editBtn);
            actions.appendChild(deleteBtn);
            
            row.appendChild(info);
            row.appendChild(actions);
            documentList.appendChild(row);
        });
    }

    // Open Document Editor (New or Edit)
    async function openDocEditor(filePath = '') {
        isEditing = !!filePath;
        docFilename.value = filePath;
        docFilename.disabled = isEditing; // Lock name if editing
        
        if (isEditing) {
            modalTitle.textContent = 'Edit Document';
            docContent.value = 'Loading content...';
            editorModal.classList.remove('hidden');
            
            try {
                const directory = dataDirectoryInput.value.trim() || './data';
                const response = await fetch(`/api/documents/read?directory=${encodeURIComponent(directory)}&path=${encodeURIComponent(filePath)}`);
                if (!response.ok) throw new Error('Could not read document');
                const data = await response.json();
                docContent.value = data.content;
            } catch (error) {
                console.error(error);
                alert(`Error loading file content: ${error.message}`);
                editorModal.classList.add('hidden');
            }
        } else {
            modalTitle.textContent = 'Create New Document';
            docContent.value = '';
            editorModal.classList.remove('hidden');
        }
    }

    // Save & Ingest Document
    btnSaveDoc.addEventListener('click', async () => {
        const path = docFilename.value.trim();
        const content = docContent.value;
        const directory = dataDirectoryInput.value.trim() || './data';

        if (!path) {
            alert('Please specify a filename/path.');
            return;
        }

        btnSaveDoc.disabled = true;
        saveSpinner.classList.remove('hidden');

        try {
            // Save document
            const saveRes = await fetch('/api/documents/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory, path, content })
            });

            if (!saveRes.ok) throw new Error(await saveRes.text() || 'Failed to save document');
            
            // Auto trigger Ingestion to update the Vector Store
            const ingestRes = await fetch('/api/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory })
            });

            if (!ingestRes.ok) throw new Error(await ingestRes.text() || 'Failed to ingest changes');
            
            editorModal.classList.add('hidden');
            await init();
        } catch (error) {
            console.error('Save error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            btnSaveDoc.disabled = false;
            saveSpinner.classList.add('hidden');
        }
    });

    // Delete Document
    async function deleteDoc(filePath) {
        if (!confirm(`Are you sure you want to delete ${filePath}? This will remove it from the disk and re-index the database.`)) {
            return;
        }

        const directory = dataDirectoryInput.value.trim() || './data';
        try {
            const deleteRes = await fetch('/api/documents/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory, path: filePath })
            });

            if (!deleteRes.ok) throw new Error(await deleteRes.text() || 'Failed to delete file');
            
            // Auto trigger Ingestion to update the Vector Store
            await fetch('/api/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory })
            });

            await init();
        } catch (error) {
            console.error('Delete error:', error);
            alert(`Error: ${error.message}`);
        }
    }

    // Modal Close Actions
    const closeModal = () => editorModal.classList.add('hidden');
    closeEditor.addEventListener('click', closeModal);
    btnCancelEdit.addEventListener('click', closeModal);
    btnNewDoc.addEventListener('click', () => openDocEditor());

    // Submit Query
    btnSubmitQuery.addEventListener('click', async () => {
        const query = queryTextarea.value.trim();
        if (!query) {
            alert('Please enter a question first.');
            return;
        }

        submitQuery(query);
    });

    async function submitQuery(query) {
        queryTextarea.value = query;
        // Set UI loading state
        btnSubmitQuery.disabled = true;
        querySpinner.classList.remove('hidden');
        answerOutput.innerHTML = '<div class="empty-state"><p>Searching vector database & generating response...</p></div>';
        retrievedContextsList.innerHTML = '<div class="empty-state"><p>Retrieving documents...</p></div>';

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topKSelect.value)
                })
            });

            if (!response.ok) {
                throw new Error(await response.text() || 'Failed to query RAG server');
            }

            const data = await response.json();
            
            // Render Answer
            answerOutput.textContent = data.answer;
            
            // Render TF-IDF token matches
            if (data.token_matches && data.token_matches.length > 0) {
                tfidfInspector.classList.remove('hidden');
                tokenMatchesList.innerHTML = '';
                data.token_matches.forEach(item => {
                    const badge = document.createElement('div');
                    badge.className = 'token-badge';
                    
                    const tokenLabel = document.createElement('span');
                    tokenLabel.className = 'token-name';
                    tokenLabel.textContent = item.word;
                    
                    const score = document.createElement('span');
                    score.className = 'token-score';
                    score.textContent = `tf-idf: ${item.tfidf.toFixed(3)} (idf: ${item.idf.toFixed(1)})`;
                    
                    badge.appendChild(tokenLabel);
                    badge.appendChild(score);
                    tokenMatchesList.appendChild(badge);
                });
            } else {
                tfidfInspector.classList.add('hidden');
            }

            // Render Retrieved Contexts
            if (data.contexts && data.contexts.length > 0) {
                retrievedContextsList.innerHTML = '';
                data.contexts.forEach(ctx => {
                    const card = document.createElement('div');
                    card.className = 'context-card';
                    
                    const meta = document.createElement('div');
                    meta.className = 'context-meta';
                    
                    const source = document.createElement('span');
                    source.className = 'source-badge';
                    source.textContent = ctx.source;
                    source.onclick = () => openDocEditor(ctx.source);
                    source.style.cursor = 'pointer';
                    source.title = "Click to view full document";
                    
                    const sim = document.createElement('span');
                    sim.className = 'similarity-badge';
                    sim.textContent = `${(ctx.similarity * 100).toFixed(1)}% Match`;
                    
                    meta.appendChild(source);
                    meta.appendChild(sim);
                    
                    const text = document.createElement('p');
                    text.className = 'context-text';
                    text.textContent = ctx.text;
                    
                    card.appendChild(meta);
                    card.appendChild(text);
                    retrievedContextsList.appendChild(card);
                });
            } else {
                retrievedContextsList.innerHTML = '<div class="empty-state"><p>No context chunks matching this query.</p></div>';
            }

            // Save to chat history
            saveToHistory(query, data.answer);

        } catch (error) {
            console.error('Query error:', error);
            answerOutput.innerHTML = `<div style="color: #ef4444; font-weight: 500;">Error: ${error.message}</div>`;
            retrievedContextsList.innerHTML = '<div class="empty-state"><p>Retrieval failed.</p></div>';
            tfidfInspector.classList.add('hidden');
        } finally {
            btnSubmitQuery.disabled = false;
            querySpinner.classList.add('hidden');
        }
    }

    // Ingest Directory
    btnIngest.addEventListener('click', async () => {
        const directory = dataDirectoryInput.value.trim();
        if (!directory) {
            alert('Please specify a valid directory path.');
            return;
        }

        btnIngest.disabled = true;
        ingestSpinner.classList.remove('hidden');
        
        try {
            const response = await fetch('/api/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ directory: directory })
            });

            if (!response.ok) {
                throw new Error(await response.text() || 'Ingestion failed');
            }

            const data = await response.json();
            alert(`Ingestion completed successfully!\n${data.message}`);
            await init();
        } catch (error) {
            console.error('Ingestion error:', error);
            alert(`Error during ingestion: ${error.message}`);
        } finally {
            btnIngest.disabled = false;
            ingestSpinner.classList.add('hidden');
        }
    });

    // Suggested Queries Pills
    suggestionPills.addEventListener('click', (e) => {
        const query = e.target.getAttribute('data-query');
        if (query) {
            submitQuery(query);
        }
    });

    // Chat History Management
    function saveToHistory(query, answer) {
        // Keep last 15 queries
        chatHistory.unshift({ query, answer, timestamp: new Date().toLocaleTimeString() });
        if (chatHistory.length > 15) chatHistory.pop();
        localStorage.setItem('acme_rag_chat_history', JSON.stringify(chatHistory));
        renderChatHistory();
    }

    function renderChatHistory() {
        if (!chatHistory || chatHistory.length === 0) {
            chatHistoryList.innerHTML = '<div class="empty-state">No chat history in this session yet.</div>';
            return;
        }

        chatHistoryList.innerHTML = '';
        chatHistory.forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const header = document.createElement('div');
            header.className = 'history-item-header';
            
            const q = document.createElement('span');
            q.className = 'history-q';
            q.textContent = item.query;
            q.onclick = () => submitQuery(item.query);
            
            const time = document.createElement('span');
            time.className = 'history-time';
            time.textContent = item.timestamp;
            
            header.appendChild(q);
            header.appendChild(time);
            
            const ans = document.createElement('div');
            ans.className = 'history-ans hidden';
            ans.textContent = item.answer;
            
            // Toggle answer visibility on click
            q.title = "Click to run query again. Click time to toggle response view.";
            time.title = "Toggle cached response";
            time.style.cursor = 'pointer';
            time.onclick = (e) => {
                e.stopPropagation();
                ans.classList.toggle('hidden');
            };

            historyItem.appendChild(header);
            historyItem.appendChild(ans);
            chatHistoryList.appendChild(historyItem);
        });
    }

    btnClearHistory.addEventListener('click', () => {
        if (confirm('Clear all conversation history?')) {
            chatHistory = [];
            localStorage.removeItem('acme_rag_chat_history');
            renderChatHistory();
        }
    });

    btnExportChat.addEventListener('click', () => {
        if (chatHistory.length === 0) {
            alert('No chat history to export.');
            return;
        }

        let mdContent = `# Acme RAG Helpline Chat Session Log\nExported on: ${new Date().toLocaleString()}\n\n---\n\n`;
        chatHistory.forEach((item, idx) => {
            mdContent += `### [${item.timestamp}] Query: ${item.query}\n\n**Response:**\n${item.answer}\n\n---\n\n`;
        });

        const blob = new Blob([mdContent], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `acme_hr_chat_log_${new Date().toISOString().slice(0,10)}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Initialize
    init();
});
