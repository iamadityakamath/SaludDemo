-- Salud CRM tables (reduced, synthetic) translated from BIG_Synthetic_Schema.sql (SQL Server) to Postgres.
-- Table and column names are unchanged, so moving back to Salud's SQL Server is a type swap:
--   uniqueidentifier -> uuid, nvarchar(n)/varchar(n) -> varchar(n), nvarchar(max) -> text,
--   money -> numeric(19,4), datetime -> timestamp, bit -> boolean, decimal -> numeric.
-- Quoted identifiers keep Salud's mixed-case names (e.g. "salud_gphpvaccountsId").
-- Synthetic data only: every GUID, date, amount and note is invented by the BIG generator.

CREATE SCHEMA IF NOT EXISTS big_synthetic;

-- One account/encounter; current account state, not a historical snapshot series.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_gphpvaccounts" (
    "salud_gphpvaccountsId" uuid NOT NULL,
    "salud_GuarantorId" uuid NULL,
    "salud_HBPB" int NULL,
    "salud_FinancialClass" varchar(100) NULL,
    "salud_PlacementStatus" int NULL,
    "salud_PlacementDate" timestamp NULL,
    "salud_LastPlacementDate" timestamp NULL,
    "salud_DischargeDate" timestamp NULL,
    "salud_PlacementPatientBalance" numeric(19,4) NULL,
    "salud_SelfPayBalance" numeric(19,4) NULL,
    "salud_SelfPayReceipts" numeric(19,4) NULL,
    "salud_SelfPayAdjustments" numeric(19,4) NULL,
    "salud_SelfPayBalanceDate" timestamp NULL,
    "salud_OnPaymentPlan" int NULL,
    "salud_EpicPaymentPlanNumberofPayments" numeric(23,10) NULL,
    "salud_EpicPaymentPlanPaymentsMissed" numeric(23,10) NULL,
    "salud_EpicPaymentPlanLastPaymentDate" timestamp NULL,
    "salud_StatementCount" int NULL,
    "salud_FirstStatementDate" timestamp NULL,
    "salud_LastStatementDate" timestamp NULL,
    "salud_LastCallDate" timestamp NULL,
    "salud_LastAccountContact" timestamp NULL,
    "salud_LastPatientPaymentFromTxnsDate" timestamp NULL,
    "salud_LastUpdatedfromClientSystem" timestamp NULL,
    PRIMARY KEY ("salud_gphpvaccountsId"),
    CHECK ("salud_HBPB" IS NULL OR "salud_HBPB" IN (785570000, 785570001)),
    CHECK ("salud_PlacementStatus" IS NULL OR "salud_PlacementStatus" IN (785570000, 785570001, 785570002)),
    CHECK ("salud_OnPaymentPlan" IS NULL OR "salud_OnPaymentPlan" IN (785570000, 785570001))
);

-- One contact/activity; may relate to multiple accounts. Includes non-call activity.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_gphpvcontacthistory" (
    "salud_gphpvcontacthistoryId" uuid NOT NULL,
    "salud_GuarantorId" uuid NULL,
    "salud_DateandTimeofContact" timestamp NULL,
    "salud_ConnectionType" int NULL,
    "salud_Contact_Type" int NULL,
    "salud_CallDisposition" int NULL,
    "salud_CollectionAttempt" int NULL,
    "salud_CollectionResult" int NULL,
    "salud_ActionsTaken" varchar(1700) NULL,
    "salud_PaymentAmount" numeric(19,4) NULL,
    "salud_ReviewBalance" boolean NULL,
    "salud_inquireaboutfinancialassistance" boolean NULL,
    "salud_RequestedCallback" boolean NULL,
    "salud_CallbackDate" timestamp NULL,
    "salud_needsomeoneelsetransfer" boolean NULL,
    "salud_TransferredTo" int NULL,
    "salud_PreferredLanguage" int NULL,
    "salud_AgentCallTalkDuration" numeric(23,10) NULL,
    "salud_AgentCallWrapUpDuration" numeric(23,10) NULL,
    "salud_Agent" varchar(200) NULL,
    "salud_TCNRecordID" varchar(100) NULL,
    "salud_Note" text NULL,
    PRIMARY KEY ("salud_gphpvcontacthistoryId"),
    CHECK ("salud_ConnectionType" IS NULL OR "salud_ConnectionType" IN (785570000, 785570001, 785570002, 785570003, 785570004, 785570005, 785570006, 785570007, 785570008, 785570009)),
    CHECK ("salud_Contact_Type" IS NULL OR "salud_Contact_Type" IN (785570000, 785570001, 785570002, 785570003, 785570004, 785570005)),
    CHECK ("salud_CallDisposition" IS NULL OR "salud_CallDisposition" IN (100000000, 100000001, 100000002, 100000003, 100000004, 100000005, 100000006, 100000007, 100000008, 100000009, 100000010, 100000011, 100000012, 100000013)),
    CHECK ("salud_CollectionAttempt" IS NULL OR "salud_CollectionAttempt" IN (785570000, 785570001)),
    CHECK ("salud_CollectionResult" IS NULL OR "salud_CollectionResult" IN (785570000, 785570001, 785570002, 785570003, 785570004, 785570005, 785570006, 785570007, 785570008, 785570009, 785570010, 785570011, 785570012)),
    CHECK ("salud_TransferredTo" IS NULL OR "salud_TransferredTo" IN (785570000, 785570001, 785570002, 785570003, 785570004, 785570005)),
    CHECK ("salud_PreferredLanguage" IS NULL OR "salud_PreferredLanguage" IN (785570000, 785570001, 785570002))
);

-- One receipt/adjustment record; account linkage is salud_ReceiptsId.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_gphpvreceipts" (
    "salud_gphpvreceiptsId" uuid NOT NULL,
    "salud_ReceiptsId" uuid NULL,
    "salud_Guarantor" uuid NULL,
    "salud_PostDate" timestamp NULL,
    "salud_Amount" numeric(19,4) NULL,
    "salud_TXNType" int NULL,
    "salud_PaymentType" int NULL,
    "salud_ExcludeTransaction" int NULL,
    "salud_HBPB" int NULL,
    "salud_LastOutboundCall" uuid NULL,
    PRIMARY KEY ("salud_gphpvreceiptsId"),
    CHECK ("salud_TXNType" IS NULL OR "salud_TXNType" IN (785570000, 785570001)),
    CHECK ("salud_PaymentType" IS NULL OR "salud_PaymentType" IN (785570000, 785570001)),
    CHECK ("salud_ExcludeTransaction" IS NULL OR "salud_ExcludeTransaction" IN (785570000, 785570001)),
    CHECK ("salud_HBPB" IS NULL OR "salud_HBPB" IN (785570000, 785570001))
);

-- One unique account/contact pair. Required to represent the many-to-many relationship.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_Account_X_ContactHistory" (
    "salud_gphpvaccountsid" uuid NOT NULL,
    "salud_gphpvcontacthistoryid" uuid NOT NULL,
    PRIMARY KEY ("salud_gphpvaccountsid", "salud_gphpvcontacthistoryid")
);

-- One synthetic guarantor key with optional outreach restriction. No identity fields.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_gphpvguarantor" (
    "salud_gphpvguarantorId" uuid NOT NULL,
    "salud_DoNotCall" int NULL,
    PRIMARY KEY ("salud_gphpvguarantorId"),
    CHECK ("salud_DoNotCall" IS NULL OR "salud_DoNotCall" IN (785570000, 785570001))
);

-- One QA review linked to one contact; multiple reviews per contact are possible.
CREATE TABLE IF NOT EXISTS big_synthetic."salud_gphpvqualityreview" (
    "salud_gphpvqualityreviewId" uuid NOT NULL,
    "salud_QualityReviewId" uuid NULL,
    "salud_reviewdate" timestamp NULL,
    "salud_AppropriateGreetings" int NULL,
    "salud_AppropriateGreetingsComment" text NULL,
    "salud_UpholdingCulture" int NULL,
    "salud_UpholdingCultureComments" text NULL,
    "salud_CapturingData" int NULL,
    "salud_CapturingDataComments" text NULL,
    "salud_UnderstandingCustomerNeeds" int NULL,
    "salud_UnderstandingCustomerNeedsComments" text NULL,
    "salud_CorrectInformation" int NULL,
    "salud_CorrectInformationComments" text NULL,
    "salud_FirstContactResolution" int NULL,
    "salud_FirstContactResolutionComments" text NULL,
    "salud_Compliance" int NULL,
    "salud_ComplianceComments" text NULL,
    "salud_CorrectCallStatusing" int NULL,
    "salud_CorrectCallStatusingComments" text NULL,
    "salud_Grammar" int NULL,
    "salud_GrammarComments" text NULL,
    "salud_lessonslearned" text NULL,
    PRIMARY KEY ("salud_gphpvqualityreviewId"),
    CHECK ("salud_AppropriateGreetings" IS NULL OR "salud_AppropriateGreetings" IN (785570000, 785570001)),
    CHECK ("salud_UpholdingCulture" IS NULL OR "salud_UpholdingCulture" IN (785570000, 785570001)),
    CHECK ("salud_CapturingData" IS NULL OR "salud_CapturingData" IN (785570000, 785570001)),
    CHECK ("salud_UnderstandingCustomerNeeds" IS NULL OR "salud_UnderstandingCustomerNeeds" IN (785570000, 785570001)),
    CHECK ("salud_CorrectInformation" IS NULL OR "salud_CorrectInformation" IN (785570000, 785570001)),
    CHECK ("salud_FirstContactResolution" IS NULL OR "salud_FirstContactResolution" IN (785570000, 785570001)),
    CHECK ("salud_Compliance" IS NULL OR "salud_Compliance" IN (785570000, 785570001)),
    CHECK ("salud_CorrectCallStatusing" IS NULL OR "salud_CorrectCallStatusing" IN (785570000, 785570001)),
    CHECK ("salud_Grammar" IS NULL OR "salud_Grammar" IN (785570000, 785570001))
);

-- Foreign keys (as in the source script)
ALTER TABLE big_synthetic."salud_Account_X_ContactHistory" DROP CONSTRAINT IF EXISTS "FK_BIG_01";
ALTER TABLE big_synthetic."salud_Account_X_ContactHistory" ADD CONSTRAINT "FK_BIG_01" FOREIGN KEY ("salud_gphpvaccountsid") REFERENCES big_synthetic."salud_gphpvaccounts" ("salud_gphpvaccountsId");
ALTER TABLE big_synthetic."salud_Account_X_ContactHistory" DROP CONSTRAINT IF EXISTS "FK_BIG_02";
ALTER TABLE big_synthetic."salud_Account_X_ContactHistory" ADD CONSTRAINT "FK_BIG_02" FOREIGN KEY ("salud_gphpvcontacthistoryid") REFERENCES big_synthetic."salud_gphpvcontacthistory" ("salud_gphpvcontacthistoryId");
ALTER TABLE big_synthetic."salud_gphpvaccounts" DROP CONSTRAINT IF EXISTS "FK_BIG_03";
ALTER TABLE big_synthetic."salud_gphpvaccounts" ADD CONSTRAINT "FK_BIG_03" FOREIGN KEY ("salud_GuarantorId") REFERENCES big_synthetic."salud_gphpvguarantor" ("salud_gphpvguarantorId");
ALTER TABLE big_synthetic."salud_gphpvcontacthistory" DROP CONSTRAINT IF EXISTS "FK_BIG_04";
ALTER TABLE big_synthetic."salud_gphpvcontacthistory" ADD CONSTRAINT "FK_BIG_04" FOREIGN KEY ("salud_GuarantorId") REFERENCES big_synthetic."salud_gphpvguarantor" ("salud_gphpvguarantorId");
ALTER TABLE big_synthetic."salud_gphpvqualityreview" DROP CONSTRAINT IF EXISTS "FK_BIG_05";
ALTER TABLE big_synthetic."salud_gphpvqualityreview" ADD CONSTRAINT "FK_BIG_05" FOREIGN KEY ("salud_QualityReviewId") REFERENCES big_synthetic."salud_gphpvcontacthistory" ("salud_gphpvcontacthistoryId");
ALTER TABLE big_synthetic."salud_gphpvreceipts" DROP CONSTRAINT IF EXISTS "FK_BIG_06";
ALTER TABLE big_synthetic."salud_gphpvreceipts" ADD CONSTRAINT "FK_BIG_06" FOREIGN KEY ("salud_ReceiptsId") REFERENCES big_synthetic."salud_gphpvaccounts" ("salud_gphpvaccountsId");
ALTER TABLE big_synthetic."salud_gphpvreceipts" DROP CONSTRAINT IF EXISTS "FK_BIG_07";
ALTER TABLE big_synthetic."salud_gphpvreceipts" ADD CONSTRAINT "FK_BIG_07" FOREIGN KEY ("salud_LastOutboundCall") REFERENCES big_synthetic."salud_gphpvcontacthistory" ("salud_gphpvcontacthistoryId");
ALTER TABLE big_synthetic."salud_gphpvreceipts" DROP CONSTRAINT IF EXISTS "FK_BIG_08";
ALTER TABLE big_synthetic."salud_gphpvreceipts" ADD CONSTRAINT "FK_BIG_08" FOREIGN KEY ("salud_Guarantor") REFERENCES big_synthetic."salud_gphpvguarantor" ("salud_gphpvguarantorId");
