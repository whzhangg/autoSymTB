"""Module containing classes to create and manipulate SISSO input files."""

import dataclasses


@dataclasses.dataclass
class KwdArg:
    """helper class to write each individual keyword"""
    name: str
    allowed_types: tuple
    comments: str

    def format_kw(self, val, float_format=".12f") -> str:
        # Determine the type of the value for this keyword
        val_type = None
        for allowed_type in self.allowed_types:
            if allowed_type is int:
                if type(val) is int:
                    val_type = int
                    break
            elif allowed_type is float:
                if type(val) is float:  # pragma: no branch
                    val_type = float
                    break
            elif allowed_type is bool:
                if type(val) is bool:  # pragma: no branch
                    val_type = bool
                    break
            elif allowed_type is str:
                if type(val) is str:  # pragma: no branch
                    val_type = str
                    break
            elif allowed_type == "list_of_ints":
                if (  # pragma: no branch
                    type(val) is list or type(val) is tuple
                ) and all([type(item) is int for item in val]):
                    val_type = "list_of_ints"
                    break
            # TODO: add checks on the str_operators, str_dimensions and str_isconvex
            elif allowed_type == "str_operators":
                val_type = "str_operators"
            elif allowed_type == "str_dimensions":  # pragma: no cover
                val_type = "str_dimensions"
            elif allowed_type == "str_isconvex":  # pragma: no cover
                val_type = "str_isconvex"

        if val_type is None:  # pragma: no cover
            raise ValueError(
                'Type of value "{}" for keyword "{}" not found/valid.'.format(
                    str(val), self.name
                )
            )

        if val_type is int:
            field = "{}={:d}".format(self.name, val)
        elif val_type is float:
            float_ref_str = "{}={{:{}}}".format(self.name, float_format)
            field = float_ref_str.format(val)
        elif val_type is bool:
            field = "{}=.{}.".format(self.name, str(val).lower())
        elif val_type is str:
            field = "{}='{}'".format(self.name, val)
        elif val_type == "list_of_ints":
            if self.name in ["subs_sis", "nsample"]:
                field = "{}={}".format(self.name, ",".join(["{:d}".format(v) for v in val]))
            else:  # pragma: no cover
                field = "{}=({})".format(self.name, ",".join(["{:d}".format(v) for v in val]))
        elif val_type == "str_operators":
            field = "{}='{}'".format(self.name, val)
        elif val_type in ["str_dimensions", "str_isconvex"]:  # pragma: no cover
            field = "{}={}".format(self.name, val)
        else:  # pragma: no cover
            raise ValueError(
                "Wrong type for SISSO value.\nSISSO keyword : {}\n"
                "Value : {} (type : {})".format(self.name, str(val), val_type)
            )

        return f"{field:<30s}! {self.comments}"


class SISSOIn:
    """a sinple class to write the keywords to input file
    """

    KWS = [
        KwdArg("ptype",
            tuple([int]), 
            "Property type 1: regression, 2:classification."),
        KwdArg("ntask", 
            tuple([int]), 
            "(R&C) Multi-task learning (MTL) is invoked if >1."),
        KwdArg("task_weighting",
            tuple([int]),
              "(R) MTL 1: no weighting (tasks treated equally), 2: weighted by the # of samples."),
        KwdArg("scmt",
            tuple([bool]), 
            "(R) Sign-Constrained MTL is invoked if .true."),
        KwdArg("desc_dim",
            tuple([int]), 
            "(R&C) Dimension of the descriptor, a hyperparmaeter."),
        KwdArg("nsample",
            tuple([int, "list_of_ints"]), 
            "(R&C) Number of samples in train.dat, for classification: (n1,n2,...)"),
        KwdArg("restart",
            tuple([bool]), 
            "(R&C) 0: starts from scratch, 1: continues the job(progress in the file CONTINUE)"),
        KwdArg("nsf",
            tuple([int]), 
            "(R&C) Number of scalar features provided in the file train.dat"),
        KwdArg("ops",
            tuple(["str_operators"]), 
            "(R&C) Please customize the operators from the list shown above."),
        KwdArg("fcomplexity",
            tuple([int]), 
            "(R&C) Max. feature complexity (# of operators in a feature), integer 0 to 7."),
        KwdArg("funit",
            tuple([str]), 
            "(R&C) (n1:n2): features from n1 to n2 in the train.dat have same units"),
        KwdArg("fmax_min",
            tuple([float]), 
            "(R&C) The feature will be discarded if the max. abs. value in it is < fmax_min."),
        KwdArg("fmax_max",
            tuple([float]), 
            "(R&C) The feature will be discarded if the max. abs. value in it is > fmax_max."),
        KwdArg("nf_sis",
            tuple([int, "list_of_ints"]), 
            "(R&C) Number of features in each of the SIS-selected subspace."),
        KwdArg("method_so",
            tuple([str]),
              "(R&C) 'L0' or 'L1L0'(LASSO+L0). The 'L0' is recommended for both ptype=1 and 2."),
        KwdArg("nl1l0",
            tuple([int]), 
            "(R) for method_so = 'L1L0', number of LASSO-selected features for the L0."),
        KwdArg("L1_max_iter",
            tuple([int]), 
            "LASSO: maximum number of iteration"),
        KwdArg("L1_dens",
            tuple([int]), 
            "LASSO: number of points to sample the penalty parameter"),
        KwdArg("L1_tole",
            tuple([float]), 
            "LASSO: tolerance for the optimization with a given lambda"),
        KwdArg("L1_minrmse",
            tuple([float]), 
            "LASSO: condition for LASSO to stop"),
        KwdArg("L1_warm_start",
            tuple([bool]), 
            "LASSO: use solution of last lambda to start new optimization"),
        KwdArg("fit_intercept",
            tuple([bool]), 
            "(R) Fit to a nonzero (.true.) or zero (.false.) intercept for the linear model."),
        KwdArg("metric",
            tuple([str]), 
            "(R) The metric for model selection in regression: RMSE or MaxAE (max abs err)"),
        KwdArg("nmodels",
            tuple([int]), 
            "(R&C) Number of the top-ranked models to output (see the folder 'models')"),
        KwdArg("isconvex",
            tuple(["str_isconvex"]), 
            "(C) Each data group constrained to be convex domain, 1: YES; 0: NO"),
        KwdArg("bwidth",
            tuple([float]), 
            "(C) Boundary tolerance for classification"),       
    ]

    def __init__(self, **keyword_arguments) -> None:
        """save all the input parameters"""
        self.all_keyword_arguments = keyword_arguments

        all_keys = [k.name for k in self.KWS]
        for arg in self.all_keyword_arguments.keys():
            if arg not in all_keys:
                raise ValueError(f"input keyword {arg} is not a valid input parameter")
        

    def input_string(self) -> str:
        """Input string of the SISSO.in file."""
        formatted_lines = []
        for kw in self.KWS:
            name = kw.name
            if name in self.all_keyword_arguments:
                formatted_lines.append(
                    kw.format_kw(self.all_keyword_arguments[name]))
        return "\n".join(formatted_lines)


    def to_file(self, filename="SISSO.in") -> None:
        """Write SISSOIn object to file."""
        with open(filename, "w") as f:
            f.write("!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            f.write("! Below are the list of keywords for SISSO. Use exclamation mark,!,to comment out a line.\n")
            f.write("! The (R), (C) and (R&C) denotes the keyword to be used by regression, classification and both, respectively.\n")
            f.write("! Users need to change the setting below according to your data and job.\n")
            f.write("!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            f.write(self.input_string())
