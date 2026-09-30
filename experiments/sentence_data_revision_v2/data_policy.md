# Europe PMC sentence data revision v2: policy fixed before revised scoring

This is a new, exploratory data version using only the 120 saved 2021 XML files and
20 saved pilot XML files. Article identity and order remain fixed. Inclusion is based
only on XML structure, original prose and the rules below; A/B/D/E NLLs and generated
text have no role in inclusion. Excluded articles are not replaced. This policy does
not modify the original selections or scores.

## Narrative start and document order

1. Work within the XML `article/body` tree, in document order. The narrative is the
   first ordinary prose paragraph (`p`) in the body or its narrative `sec` descendants.
   Section titles, displayed quotations (`disp-quote`), abstract/front matter, author
   material, reference lists, back matter, captions, tables, figures,
   acknowledgments and supplementary material are
   not prose paragraphs. A clearly labeled specifications/method metadata table,
   even when represented as `body/p`, is not prose; begin after the entire metadata
   block. Do not omit an earlier qualifying prose sentence merely because its
   punctuation, citation or capitalization is difficult.
2. A sentence-bearing `boxed-text` or `list` before the first ordinary prose
   paragraph makes the narrative start ambiguous: exclude that article instead of
   silently choosing one container. Apply this rule to all 140 XMLs. In particular,
   PMC8236113, PMC8840875 and PMC9035235 are excluded if their saved XML confirms
   the already observed sentence-bearing prelude. Later boxed text/lists are ignored
   only if encountered after six valid narrative sentences; if encountered among the
   first six, exclude unless they are demonstrably non-narrative apparatus.
3. Do not use headings, titles, author strings, figure/table captions, reference
   entries or bibliographic/specification fields as sentences. A paragraph containing
   both metadata and prose must have a structurally clear boundary; otherwise
   exclude. Preserve the original order of qualifying prose paragraphs. Sentences
   may cross paragraph boundaries only when the earlier paragraph demonstrably
   continues; uncertain continuation causes exclusion.

## Text fidelity and sentence boundaries

4. Flatten ordinary inline markup in place, retaining its visible text and order.
   Retain visible citation and `xref` labels, including numeric reference markers
   and `Figure 1A`/`Figure 1B`; remove only markup, not their printed text. Normalize
   XML whitespace runs to one space. Do not paraphrase, add missing scientific
   content or manufacture a smooth sentence. The six saved sentences must be
   reconstructible as ordered substrings of this normalized XML prose after only
   documented whitespace normalization.
5. Treat citation labels after terminal punctuation as part of the preceding sentence
   (`sentence.[1–3] Next` splits before `Next`; likewise `sentence.1–3 Next` and
   numeric superscript citation after a period). A
   trailing citation comma or dash is not a reason to discard the sentence. Preserve
   the citation text in the sentence and record XML position.
6. Protect periods inside a personal initial before a surname or additional initials
   (`F. Krammer`, `G. Forni`), known titles and place abbreviations (`St. Louis`),
   `syn.`, `sp.`, `spp.`, `et al.`, `e.g.`, `i.e.`, `Dr.`, `Prof.`, `Fig.`, `Eq.`,
   `U.S.`, `U.K.`, Latin taxonomic author abbreviations (`Phlomis L.` followed by
   qualifying text), decimal numbers and numbered references. Protect abbreviations
   based on their local grammatical context, not simply a global list that would
   hide a real sentence ending. A capitalized new sentence and an identifiable
   lowercase-starting technical term (`fNIRS`, `miR172`, etc.) can begin after a true
   terminal mark. Such boundaries require review against the XML context.
7. A sentence ends only at genuine terminal `.`, `?` or `!`, optionally followed by
   closing quote/parenthesis and citation marker, with an actual next sentence or
   paragraph boundary. A period in an abbreviation or decimal is not terminal.
   Preserve quotes and brackets. A short or long sentence is not excluded on length
   alone. All six must be complete, consecutive sentences from the chosen narrative
   start; any unresolved boundary among them excludes the article.
8. Inline and display formulas count as source content. If a formula in or before
   the first six narrative sentences cannot be faithfully represented from visible
   XML text (including MathML/TeX where necessary), or its placement makes a
   sentence boundary uncertain, exclude the article. Never delete it and then score
   an apparently fluent substitute. Preserve figure cross-reference labels when
   they are visible; exclude if an essential label cannot be recovered.

## Review and lock gate

9. For each XML, record its SHA-256, six complete sentence strings and source
   paragraph/XPath spans, the fifth/sixth boundary context, comparison with the
   original prompt and target, and an explicit include/exclude decision with reason.
   Human review covers every one of the 140 six-sentence windows, including the 46
   previously pending and 72 previously passed. Review all revised or excluded
   windows in their raw XML context, especially initials, botanical authors,
   citations, lowercase technical terms, equations and figure references.
10. An article is included only if the first six sentences and narrative start are
    unambiguous and faithful to the saved XML. Mark every uncertain item excluded
    with its unresolved reason; do not promote it to a pass. Lock two separate
    manifests and their SHA-256 before tokenizer preflight or model loading. Both
    included and excluded identities stay in the per-article decision record.
11. Rebuild the old system/user Qwen chat prompt from the revised first five sentences.
    Score the full sixth sentence with **one shared target ID sequence encoded by
    the official A tokenizer**, irrespective of A/B/D/E prompt tokenizer. This is
    conditional probability for a fixed ID continuation, not each condition's
    naturally re-tokenized full text. Re-score every included article; old NLLs are
    never reused. Report 2021 and pilot separately and label all revised statistics
    exploratory, not the old locked main result or MAUVE.
