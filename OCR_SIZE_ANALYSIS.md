# OCR Size Analysis - Key Findings

## Critical Discovery: Document AI Response Size is TINY!

### The Real Numbers
**Total for 30 pages:**
- Input (page_bytes): 6.2 MB
- Document AI Response: **0.1 MB** (86.8 KB total)
- Average response per page: **2.9 KB**

**This means Document AI responses are NOT the memory problem!**

## Analysis by Page Type

### Small Pages (1-3)
- DPI=45
- Input: ~39 KB
- Response: **0.9-1.1 KB**
- Very minimal impact

### Large Pages - LOW Complexity (Pages 4-12)
At DPI=30:
- Input: 116-331 KB
- Response: **1.0-4.9 KB**
- Maximum: 4.9 KB (page 6)

### Large Pages - MEDIUM Complexity (Pages 13-23)
At DPI=30:
- Input: 145-363 KB
- Response: **1.3-10.8 KB**
- Page 14 (complex diagram): 10.8 KB response
- Still very small!

### Large Pages - HIGH Complexity (Pages 25-29)
At DPI=30:
- Input: 201-365 KB
- Response: **2.2-11.7 KB**
- Page 29 (most complex): 11.7 KB response
- Even the "worst" page is only 11.7 KB!

## DPI Comparison for Large Pages

### Page 14 (complex diagram - 2013 OCR elements):
- DPI=20: Input=190 KB, Response=0.4 KB
- DPI=25: Input=264 KB, Response=**12.7 KB**
- DPI=30: Input=364 KB, Response=**10.8 KB**

### Page 29 (very complex - 1879 OCR elements):
- DPI=20: Input=186 KB, Response=0.4 KB  
- DPI=25: Input=261 KB, Response=**11.7 KB**
- DPI=30: Input=357 KB, Response=**11.4 KB**

## CRITICAL INSIGHT

**The Document AI response is NOT causing the 350MB memory spike!**

The responses are tiny (< 12 KB even for the most complex pages). Something else is holding onto memory.

## What IS Causing the Memory Problem?

Given this data, the memory issue must be from:

1. ❌ **NOT Document AI responses** (< 12 KB each)
2. ❌ **NOT page_bytes input** (< 400 KB each at DPI=30)
3. ✅ **Likely: Python objects/caching in memory**
   - `page_bytes` not being freed properly
   - LLM instances/chat history
   - PyMuPDF document handles
   - Session state accumulation
   
4. ✅ **Likely: Garbage collection delays**
   - Memory allocated but not freed
   - Objects waiting in GC queue
   - Reference cycles

## Recommendation

**DO NOT reduce DPI further!** The DPI=30 responses are already tiny (< 12 KB). 

Instead, focus on:
1. More aggressive garbage collection BEFORE each large page
2. Explicit deletion of intermediate objects  
3. Force GC between document operations
4. Clear Python caches more frequently

## Memory Budget Reality Check

For 84 pages with current approach:
- Total page_bytes: ~15-20 MB (not the problem)
- Total DocAI responses: < 1 MB (not the problem)
- **Mystery memory**: ~300-400 MB (THIS is the problem!)

The memory is being held by Python internals, not our OCR data!

