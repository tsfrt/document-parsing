-- =============================================================================
-- ai-parse-doc :: example queries
-- Replace `main.doc_parsing` with your `${catalog}.${schema}` if different.
-- =============================================================================

-- 1) Parse all PDFs/images in a Volume with a single model.
SELECT path,
       main.doc_parsing.parse_doc_florence(content) AS markdown
FROM read_files(
  '/Volumes/main/doc_parsing/inbox/',
  format => 'binaryFile',
  fileNamePattern => '*.{pdf,png,jpg,jpeg,PDF,PNG,JPG,JPEG}'
)
WHERE _metadata.file_size < 25 * 1024 * 1024;  -- 25MB safety cap

-- 2) Compare two models side-by-side.
SELECT path,
       main.doc_parsing.parse_doc(content, 'florence') AS via_florence,
       main.doc_parsing.parse_doc(content, 'granite')   AS via_granite
FROM read_files('/Volumes/main/doc_parsing/inbox/', format => 'binaryFile');

-- 3) Hand-picked input via base64 (useful for ad-hoc tests).
SELECT main.doc_parsing.parse_doc_phi3(unbase64('JVBERi0xLj... <truncated> ...'));

-- 4) Direct ai_query (no UDF) for full control over modelParameters.
--    Phi-3.5-vision and Granite-Vision both honour the optional `prompt` field.
SELECT path,
       ai_query(
         'doc-parser-phi3-vision',
         named_struct(
           'image_b64', base64(content),
           'prompt',    'Extract the table on this page as Markdown only. No prose.',
           'output_format', 'markdown'
         ),
         failOnError => false
       ) AS result
FROM read_files('/Volumes/main/doc_parsing/inbox/', format => 'binaryFile');

-- 5) Materialize parsed Markdown to a Delta table.
CREATE OR REPLACE TABLE main.doc_parsing.parsed_documents AS
SELECT path,
       _metadata.file_modification_time AS modified_at,
       main.doc_parsing.parse_doc_florence(content) AS markdown
FROM read_files(
  '/Volumes/main/doc_parsing/inbox/',
  format => 'binaryFile',
  fileNamePattern => '*.{pdf,png,jpg,jpeg,PDF,PNG,JPG,JPEG}'
);

-- 6) Specialist routing: send tables / charts / forms to Granite-Vision,
--    everything else to Florence as a fast default.
SELECT path,
       CASE WHEN lower(path) LIKE '%table%' OR lower(path) LIKE '%form%' OR lower(path) LIKE '%chart%'
            THEN main.doc_parsing.parse_doc_granite(content)
            ELSE main.doc_parsing.parse_doc_florence(content)
       END AS markdown
FROM read_files('/Volumes/main/doc_parsing/inbox/', format => 'binaryFile');
