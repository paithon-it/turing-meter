# A18: one probit mixed model per ChatBench comparison (PROTOCOL.md, A18). Reads DIR/NN.csv, writes DIR/fits.json.
# Usage: Rscript angle/glmm.R DIR
suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
dir <- commandArgs(trailingOnly = TRUE)[1]
files <- sort(list.files(dir, pattern = "^[0-9]+\\.csv$", full.names = TRUE))
fits <- list()
for (f in files) {
  d <- read.csv(f, stringsAsFactors = FALSE)
  d$type <- factor(d$type, levels = c("person", "machine"))
  d$is_person <- as.numeric(d$type == "person")
  warn <- character(0)
  t0 <- Sys.time()
  m <- withCallingHandlers(
    glmer(cbind(succ, n - succ) ~ 0 + type + (0 + type | question) + (0 + is_person | worker),
          data = d, family = binomial(link = "probit"), nAGQ = 1,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") })
  vc <- VarCorr(m)$question
  sp2 <- vc["typeperson", "typeperson"]; sm2 <- vc["typemachine", "typemachine"]; cv <- vc["typeperson", "typemachine"]
  i <- as.integer(sub("\\.csv$", "", basename(f)))
  fits[[length(fits) + 1]] <- list(
    i = i, sigma_p2 = sp2, sigma_m2 = sm2, cov = cv, rho_c = cv / sqrt((1 + sp2) * (1 + sm2)),
    corr_question = attr(vc, "correlation")["typeperson", "typemachine"],
    sigma_worker2 = VarCorr(m)$worker[1, 1], beta = as.list(fixef(m)), loglik = as.numeric(logLik(m)),
    singular = isSingular(m), warnings = warn, messages = m@optinfo$conv$lme4$messages,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  cat(sprintf("%02d rho_c %.4f sp2 %.3f sm2 %.3f singular %s warnings %d (%.0f s)\n", i, cv / sqrt((1 + sp2) * (1 + sm2)),
              sp2, sm2, isSingular(m), length(warn), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
writeLines(toJSON(fits, auto_unbox = TRUE, digits = NA, null = "null"), file.path(dir, "fits.json"))
cat("R", R.version.string, "lme4", as.character(packageVersion("lme4")), "\n")
