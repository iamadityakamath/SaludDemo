-- The "final" package Salud sent on Nov 17, 2026 (data/BIG_Final_Package_84, gitignored): real, redacted CRM
-- rows for 84 calls plus their transcripts. Loaded by `salud-db load-final`, which applies only this file.
--
-- Stored as Salud sent it, one table per workbook tab: column names unchanged, option fields kept as their
-- labels ('Inbound Call', 'Yes'), IDs kept with their prefixes ('Contact_<GUID>'). The Oct 14 handoff's option
-- codes are in final.option_codes for looking up a label's code.
-- Only links that hold for every row are enforced. Not enforced, because the parent isn't in the package:
-- guarantor IDs (no Guarantors tab) and transactions.salud_LastOutboundCall (mostly calls outside this extract).
-- Dates are stored as given, without a time zone.

CREATE SCHEMA IF NOT EXISTS final;
COMMENT ON SCHEMA final IS 'Salud final package (Nov 2026): real redacted CRM extract for 84 calls + transcripts, as received';

-- Tab "Accounts" (CRM salud_gphpvaccounts). salud_FinancialClass is not in this extract.
CREATE TABLE IF NOT EXISTS final.accounts (
    "salud_gphpvaccountsId" text PRIMARY KEY,
    "salud_GuarantorId" text,
    "salud_HBPB" text,
    "salud_PlacementStatus" text,
    "salud_PlacementDate" timestamp,
    "salud_LastPlacementDate" timestamp,
    "salud_DischargeDate" timestamp,
    "salud_PlacementPatientBalance" numeric(19,4),
    "salud_SelfPayBalance" numeric(19,4),
    "salud_SelfPayReceipts" numeric(19,4),
    "salud_SelfPayAdjustments" numeric(19,4),
    "salud_SelfPayBalanceDate" timestamp,
    "salud_OnPaymentPlan" text,
    "salud_EpicPaymentPlanNumberofPayments" numeric(23,10),
    "salud_EpicPaymentPlanPaymentsMissed" numeric(23,10),
    "salud_EpicPaymentPlanLastPaymentDate" timestamp,
    "salud_StatementCount" integer,
    "salud_FirstStatementDate" timestamp,
    "salud_LastStatementDate" timestamp,
    "salud_LastCallDate" timestamp,
    "salud_LastAccountContact" timestamp,
    "salud_LastPatientPaymentFromTxnsDate" timestamp,
    "salud_LastUpdatedfromClientSystem" timestamp
);

-- Tab "Contact History" (CRM salud_gphpvcontacthistory). One row per call; salud_TCNRecordID names its transcript.
CREATE TABLE IF NOT EXISTS final.contact_history (
    "salud_gphpvcontacthistoryId" text PRIMARY KEY,
    "salud_GuarantorId" text,
    "salud_DateandTimeofContact" timestamp,
    "salud_ConnectionType" text,
    "salud_Contact_Type" text,
    "salud_CallDisposition" text,
    "salud_CollectionAttempt" text,
    "salud_CollectionResult" text,
    "salud_ActionsTaken" varchar(1700),          -- several labels joined by '; '
    "salud_PaymentAmount" numeric(19,4),
    "salud_ReviewBalance" text CHECK ("salud_ReviewBalance" IN ('Yes', 'No')),
    "salud_inquireaboutfinancialassistance" text CHECK ("salud_inquireaboutfinancialassistance" IN ('Yes', 'No')),
    "salud_RequestedCallback" text CHECK ("salud_RequestedCallback" IN ('Yes', 'No')),
    "salud_CallbackDate" timestamp,
    "salud_needsomeoneelsetransfer" text CHECK ("salud_needsomeoneelsetransfer" IN ('Yes', 'No')),
    "salud_TransferredTo" text,
    "salud_PreferredLanguage" text,
    "salud_AgentCallTalkDuration" numeric(23,10),
    "salud_AgentCallWrapUpDuration" numeric(23,10),
    "salud_Agent" varchar(200),
    "salud_TCNRecordID" varchar(100) UNIQUE,
    "salud_Note" text
);

-- Tab "Account Contact Links" (CRM salud_Account_X_ContactHistory): one row per account/call pair.
CREATE TABLE IF NOT EXISTS final.account_contact_links (
    "salud_gphpvaccountsid" text NOT NULL REFERENCES final.accounts ("salud_gphpvaccountsId"),
    "salud_gphpvcontacthistoryid" text NOT NULL REFERENCES final.contact_history ("salud_gphpvcontacthistoryId"),
    PRIMARY KEY ("salud_gphpvaccountsid", "salud_gphpvcontacthistoryid")
);

-- Tab "Transactions" (CRM salud_gphpvreceipts). salud_ReceiptsId is the account key. salud_ExcludeTransaction
-- is not in this extract, so Salud's self-pay definition (which needs it) can't be applied exactly.
CREATE TABLE IF NOT EXISTS final.transactions (
    "salud_gphpvreceiptsId" text PRIMARY KEY,
    "salud_ReceiptsId" text REFERENCES final.accounts ("salud_gphpvaccountsId"),
    "salud_Guarantor" text,
    "salud_PostDate" timestamp,
    "salud_Amount" numeric(19,4),
    "salud_TXNType" text,
    "salud_PaymentType" text,
    "salud_HBPB" text,
    "salud_LastOutboundCall" text                -- not enforced: most point to calls outside this extract
);

-- Tab "QA Reviews" (CRM salud_gphpvqualityreview). salud_QualityReviewId is the call key.
CREATE TABLE IF NOT EXISTS final.qa_reviews (
    "salud_gphpvqualityreviewId" text PRIMARY KEY,
    "salud_QualityReviewId" text REFERENCES final.contact_history ("salud_gphpvcontacthistoryId"),
    "salud_reviewdate" timestamp,
    "salud_AppropriateGreetings" text CHECK ("salud_AppropriateGreetings" IN ('Yes', 'No', 'N/A')),
    "salud_AppropriateGreetingsComment" text,
    "salud_UpholdingCulture" text CHECK ("salud_UpholdingCulture" IN ('Yes', 'No', 'N/A')),
    "salud_UpholdingCultureComments" text,
    "salud_CapturingData" text CHECK ("salud_CapturingData" IN ('Yes', 'No', 'N/A')),
    "salud_CapturingDataComments" text,
    "salud_UnderstandingCustomerNeeds" text CHECK ("salud_UnderstandingCustomerNeeds" IN ('Yes', 'No', 'N/A')),
    "salud_UnderstandingCustomerNeedsComments" text,
    "salud_CorrectInformation" text CHECK ("salud_CorrectInformation" IN ('Yes', 'No', 'N/A')),
    "salud_CorrectInformationComments" text,
    "salud_FirstContactResolution" text CHECK ("salud_FirstContactResolution" IN ('Yes', 'No', 'N/A')),
    "salud_FirstContactResolutionComments" text,
    "salud_Compliance" text CHECK ("salud_Compliance" IN ('Yes', 'No', 'N/A')),
    "salud_ComplianceComments" text,
    "salud_CorrectCallStatusing" text CHECK ("salud_CorrectCallStatusing" IN ('Yes', 'No', 'N/A')),
    "salud_CorrectCallStatusingComments" text,
    "salud_Grammar" text CHECK ("salud_Grammar" IN ('Yes', 'No', 'N/A')),
    "salud_GrammarComments" text,
    "salud_lessonslearned" text
);

-- Tab "Schema Fields": Salud's description of every column above (and of tabs not sent).
CREATE TABLE IF NOT EXISTS final.schema_fields (
    data_sheet      text NOT NULL,
    entity          text NOT NULL,
    column_name     text NOT NULL,
    display_name    text,
    source_sql_type text,
    description     text,
    schema_context  text,
    PRIMARY KEY (data_sheet, column_name)
);

-- Tab "Relationships": how the tabs join, as Salud documents it.
CREATE TABLE IF NOT EXISTS final.relationships (
    child_sheet  text NOT NULL,
    child_key    text NOT NULL,
    parent_sheet text NOT NULL,
    parent_key   text NOT NULL,
    context      text,
    PRIMARY KEY (child_sheet, child_key)
);

-- Option codes from the Oct 14 schema handoff (BIG_Schema.json): label -> Salud's integer code.
CREATE TABLE IF NOT EXISTS final.option_codes (
    entity      text NOT NULL,
    field       text NOT NULL,                   -- lower case, as in the handoff
    code        integer NOT NULL,
    label       text NOT NULL,
    project_use text,
    PRIMARY KEY (entity, field, code),
    UNIQUE (entity, field, label)
);

-- The 84 transcript files, one per call. Not part of Salud's CRM schema; added by the project.
-- phase1_call_id names the older version of the same call in phase1.calls (8 calls), whose text differs.
CREATE TABLE IF NOT EXISTS final.transcripts (
    "salud_TCNRecordID" varchar(100) PRIMARY KEY REFERENCES final.contact_history ("salud_TCNRecordID"),
    file_name      text NOT NULL UNIQUE,
    transcript     text NOT NULL,                -- file text, Windows line endings made '\n'; otherwise unchanged
    turns          integer NOT NULL,
    has_caller     boolean NOT NULL,             -- false: no caller turn (voicemail / phone menu only)
    phase1_call_id text
);
