-- =============================================================================
-- ai-parse-doc :: SQL UDFs
--
-- Run this from a Databricks SQL warehouse (Pro or Serverless) on DBR 15.4+.
-- Driven by the deploy-ocr-udfs job, which substitutes :catalog and :schema
-- from job parameters before applying.
--
-- Result:
--   - One thin per-model UDF (parse_doc_florence, parse_doc_phi3,
--     parse_doc_granite) that wraps ai_query against the matching endpoint.
--   - One router UDF (parse_doc) that lets callers pick a model by string.
--
-- All UDFs accept a BINARY column (e.g. the `content` column produced by
-- read_files(..., format => 'binaryFile')) and return a STRING (markdown by
-- default, or the JSON-stringified error envelope when ai_query fails).
-- =============================================================================

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

CREATE OR REPLACE FUNCTION parse_doc_florence(content BINARY)
RETURNS STRING
COMMENT 'Florence-2 large-ft (microsoft/Florence-2-large-ft). Compact 770M VLM with task-prompted OCR; default prompt is <OCR>.'
RETURN COALESCE(
  ai_query(
    'doc-parser-florence',
    named_struct('image_b64', base64(content)),
    failOnError => false
  ).response,
  to_json(
    named_struct(
      'error',
      ai_query(
        'doc-parser-florence',
        named_struct('image_b64', base64(content)),
        failOnError => false
      ).errorMessage
    )
  )
);

CREATE OR REPLACE FUNCTION parse_doc_phi3(content BINARY)
RETURNS STRING
COMMENT 'Phi-3.5-vision-instruct (microsoft/Phi-3.5-vision-instruct). Strong general multimodal OCR/reasoning.'
RETURN COALESCE(
  ai_query(
    'doc-parser-phi3-vision',
    named_struct('image_b64', base64(content)),
    failOnError => false
  ).response,
  to_json(
    named_struct(
      'error',
      ai_query(
        'doc-parser-phi3-vision',
        named_struct('image_b64', base64(content)),
        failOnError => false
      ).errorMessage
    )
  )
);

CREATE OR REPLACE FUNCTION parse_doc_granite(content BINARY)
RETURNS STRING
COMMENT 'Granite-Vision-3.2-2B (ibm-granite/granite-vision-3.2-2b). IBM compact doc-understanding VLM.'
RETURN COALESCE(
  ai_query(
    'doc-parser-granite-vision',
    named_struct('image_b64', base64(content)),
    failOnError => false
  ).response,
  to_json(
    named_struct(
      'error',
      ai_query(
        'doc-parser-granite-vision',
        named_struct('image_b64', base64(content)),
        failOnError => false
      ).errorMessage
    )
  )
);

-- Router UDF: pick a model by string. Defaults to 'florence' if NULL.
CREATE OR REPLACE FUNCTION parse_doc(content BINARY, model STRING)
RETURNS STRING
COMMENT 'Dispatch document parsing to one of the three OCR endpoints. model in {florence, phi3, granite}.'
RETURN CASE lower(coalesce(model, 'florence'))
  WHEN 'florence' THEN parse_doc_florence(content)
  WHEN 'phi3'     THEN parse_doc_phi3(content)
  WHEN 'granite'  THEN parse_doc_granite(content)
  ELSE to_json(named_struct('error', concat('unknown model: ', coalesce(model, 'NULL'))))
END;
